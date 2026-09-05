# -*- coding: utf-8 -*-
"""
Otimizacao Terminal do Refinador Hibrido em L63 com Projecoes Geometricas Congeladas
Arquitetura:
h' = h + (h U_c) W_c + (h V_e) W_e
onde:
- U_c (r_c) e V_e (r_s) sao buffers estritamente CONGELADOS derivados da geometria UEM.
- Apenas W_c e W_e sao treinados com loss terminal (KL + 0.1 CE + 0.05 MSE) via AdamW.
Avalia:
1. Config 1: Pure Universal U_c(32) (r=32 canais)
2. Config 2: Hibrido (r_c=16, r_s=16) (r_tot=32 canais)
3. Config 3: Hibrido (r_c=32, r_s=16) (r_tot=48 canais)
Compara contra o Baseline SVD-256 Uniforme (PPL = 192.68) e L63 Monolitico (PPL = 179.94).
"""
import os
import sys
import time
import copy
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

repo_root = r"C:\Users\Nyx\Desktop\MathQwen"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, load_layer_module, dequant, get_snapshot_dir
from atlas.projection import project_layer_weights
from atlas.residual import LinearResidualStabilizer

class HybridResidualStabilizer(nn.Module):
    def __init__(self, U_c: torch.Tensor, V_e: torch.Tensor = None, alpha: float = 1.0):
        super().__init__()
        # Buffers congelados para as projecoes geometricas
        self.register_buffer("U_c", U_c.clone().to(dtype=torch.bfloat16))
        self.has_specific = (V_e is not None and V_e.shape[1] > 0)
        
        r_c = U_c.shape[1]
        self.W_c = nn.Parameter(torch.zeros(r_c, 5120, dtype=torch.bfloat16, device=U_c.device))
        
        if self.has_specific:
            self.register_buffer("V_e", V_e.clone().to(dtype=torch.bfloat16))
            r_s = V_e.shape[1]
            self.W_e = nn.Parameter(torch.zeros(r_s, 5120, dtype=torch.bfloat16, device=V_e.device))
        else:
            self.V_e = None
            self.W_e = None

        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32, device=U_c.device))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        corr_c = torch.matmul(torch.matmul(h, self.U_c), self.W_c)
        if self.has_specific:
            corr_s = torch.matmul(torch.matmul(h, self.V_e), self.W_e)
            corr = corr_c + corr_s
        else:
            corr = corr_c

        if self.alpha.item() == 1.0 and not self.alpha.requires_grad:
            return h + corr
        a = self.alpha.to(dtype=h.dtype)
        return a * h + corr

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor = None):
    vocab_size = logits.shape[-1]
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
    targets_flat = targets.reshape(-1)
    nll = F.cross_entropy(logits_pred, targets_flat, reduction="mean").item()
    ppl = math.exp(nll)
    top1 = (logits_pred.argmax(dim=-1) == targets_flat).float().mean().item() * 100.0

    cos_sim = 0.0
    kl_div = 0.0
    if teacher_logits is not None:
        t_pred = teacher_logits[:, :-1, :].reshape(-1, vocab_size).float()
        cos_sim = F.cosine_similarity(logits_pred, t_pred, dim=-1).mean().item()
        p_t = F.softmax(t_pred, dim=-1)
        log_p_s = F.log_softmax(logits_pred, dim=-1)
        kl_div = F.kl_div(log_p_s, p_t, reduction="batchmean").item()

    return {
        "nll": nll,
        "ppl": ppl,
        "top1": top1,
        "cos_sim": cos_sim,
        "kl_div": kl_div
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("OTIMIZACAO TERMINAL DO REFINADOR HIBRIDO (PROJECOES GEOMETRICAS CONGELADAS) EM L63")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Baseline Canônico SVD-256 Uniforme: PPL = 192.68")
    print("Recorde L63 Monolitico (Doc 13):      PPL = 179.94")

    # 1. Carregar Bases U_c e V_specific de L63
    bases_path = os.path.join(repo_root, "checkpoints", "common_specific_bases_l63.pt")
    if not os.path.exists(bases_path):
        raise FileNotFoundError(f"Checkpoint {bases_path} nao encontrado!")

    saved_bases = torch.load(bases_path, map_location=device)
    U_c_dict = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in saved_bases["U_c"].items()}
    V_spec_dict = {k: {d: v.to(device=device, dtype=torch.bfloat16) for d, v in d_dict.items()} for k, d_dict in saved_bases["V_specific"].items()}

    # 2. Carregar Componentes e Targets
    snapshot_dir = get_snapshot_dir()
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_train = 16
    num_test = 16
    total_seqs = num_train + num_test
    seq_len = 64

    input_ids = tokens_all[:total_seqs * seq_len].view(total_seqs, seq_len).to(device)
    train_ids = input_ids[:num_train]
    test_ids = input_ids[num_train:]
    train_targets = train_ids[:, 1:].contiguous()
    test_targets = test_ids[:, 1:].contiguous()

    comp = load_qwen_reference_components(device=device, seq_len=seq_len, num_seqs=total_seqs)
    cfg = comp["cfg"]
    lm_head_w = comp["lm_head_w"]
    norm_mod = comp["norm_mod"]
    vocab_sz = lm_head_w.shape[0]

    # Targets do Professor
    p_data = torch.load("checkpoints/prof_targets_cache.pt", map_location=device)
    logits_prof_tr = p_data["logits_prof_tr"].to(device)
    logits_prof_test = p_data["logits_prof_test"].to(device)
    prof_intermediates_tr = [x.to(device) for x in p_data["prof_intermediates_tr"]]
    y_tr_63 = prof_intermediates_tr[63].to(device).reshape(-1, 5120)

    # Entrada de L63
    h63_cache_path = os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt")
    h_in_63 = torch.load(h63_cache_path, map_location=device).to(device)

    # Forward pass da Layer 63 Atlas
    atlas_bases_path = os.path.join(repo_root, "checkpoints", "atlas_bases.pt")
    raw_bases = torch.load(atlas_bases_path, map_location=device)
    chart15_bases = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in raw_bases[15].items()}

    l63_mod = load_layer_module(63, cfg, device, snapshot_dir=snapshot_dir)
    import safetensors.torch as st
    lp = os.path.join(snapshot_dir, "layers-63.safetensors")
    with st.safe_open(lp, framework="pt") as f:
        mix_w = dequant(f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight"), f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        gate_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        up_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        down_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        cw = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": True}
    p_w = project_layer_weights(cw, chart15_bases)
    l63_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
    l63_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
    l63_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
    l63_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, total_seqs, -1)
    pos_emb = comp["rotary"](h_in_63, pos_ids)
    with torch.no_grad():
        out_s = l63_mod(h_in_63, position_embeddings=pos_emb)
        h_63_raw = (out_s[0] if isinstance(out_s, tuple) else out_s).to(torch.bfloat16)

    del l63_mod
    torch.cuda.empty_cache()

    # Baselines
    stab_base_path = os.path.join(repo_root, "checkpoints", "svd256_baseline_stabilizers.pt")
    raw_stabs = torch.load(stab_base_path, map_location=device)
    stab_base_63 = LinearResidualStabilizer(
        raw_stabs[63]["W_down"],
        raw_stabs[63]["W_up"],
        alpha=raw_stabs[63].get("alpha", 1.0)
    ).to(device=device, dtype=torch.bfloat16)
    stab_base_63.freeze()

    with torch.no_grad():
        # Raw sem refiner
        l_tr_raw = F.linear(norm_mod(h_63_raw[:num_train]), lm_head_w)
        l_te_raw = F.linear(norm_mod(h_63_raw[num_train:]), lm_head_w)
        m_raw_te = compute_metrics(l_te_raw, test_targets, teacher_logits=logits_prof_test)

        # Baseline SVD-256
        h_corr_base = stab_base_63(h_63_raw)
        l_tr_base = F.linear(norm_mod(h_corr_base[:num_train]), lm_head_w)
        l_te_base = F.linear(norm_mod(h_corr_base[num_train:]), lm_head_w)
        m_tr_base = compute_metrics(l_tr_base, train_targets)
        m_te_base = compute_metrics(l_te_base, test_targets, teacher_logits=logits_prof_test)

    print("\n" + "-" * 85)
    print(f"Atlas Raw L63 (sem refiner) : NLL={m_raw_te['nll']:.4f} | PPL={m_raw_te['ppl']:.2f} | Top-1={m_raw_te['top1']:.2f}% | KL={m_raw_te['kl_div']:.4f}")
    print(f"Baseline SVD-256 L63 (r=256) : NLL={m_te_base['nll']:.4f} | PPL={m_te_base['ppl']:.2f} | Top-1={m_te_base['top1']:.2f}% | KL={m_te_base['kl_div']:.4f}")
    print("-" * 85)

    # 3. Definir as Configuracoes Estruturais a Otimizar
    configs = [
        {
            "name": "Pure Universal U_c(32)",
            "U_c": U_c_dict[32],
            "V_e": None,
            "r_c": 32,
            "r_s": 0,
            "r_tot": 32
        },
        {
            "name": "Hibrido (r_c=16, r_s=16)",
            "U_c": U_c_dict[16],
            "V_e": V_spec_dict[16]["WikiText"][:, :16],
            "r_c": 16,
            "r_s": 16,
            "r_tot": 32
        },
        {
            "name": "Hibrido (r_c=32, r_s=16)",
            "U_c": U_c_dict[32],
            "V_e": V_spec_dict[32]["WikiText"][:, :16],
            "r_c": 32,
            "r_s": 16,
            "r_tot": 48
        }
    ]

    all_experiments = {}
    best_overall_ppl = 9999.0
    best_config_name = None

    lr = 1e-4
    num_steps = 30

    for cfg_item in configs:
        cfg_name = cfg_item["name"]
        r_tot = cfg_item["r_tot"]
        print(f"\n=========================================================================================")
        print(f"OTIMIZANDO: {cfg_name} (r_tot={r_tot} canais, U_c e V_e congelados, lr={lr:.0e})")
        print(f"=========================================================================================")

        model = HybridResidualStabilizer(cfg_item["U_c"], cfg_item["V_e"]).to(device=device, dtype=torch.bfloat16)
        
        # Otimizador apenas sobre W_c e W_e (U_c e V_e sao buffers congelados)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in trainable_params)
        print(f"  -> Parametros treinaveis: {param_count:,} ({param_count / (2*5120*256) * 100:.2f}% do SVD-256)")

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
        history = []

        for step in range(1, num_steps + 1):
            model.train()
            optimizer.zero_grad()

            h_tr_in = h_63_raw[:num_train]
            h_corr = model(h_tr_in)
            x_norm = norm_mod(h_corr)
            logits = F.linear(x_norm, lm_head_w)

            s_flat = logits[:, :-1, :].reshape(-1, vocab_sz)
            t_flat = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
            y_flat = train_targets.reshape(-1)

            loss_kl = F.kl_div(F.log_softmax(s_flat.float(), dim=-1), F.softmax(t_flat.float(), dim=-1), reduction="batchmean")
            loss_ce = F.cross_entropy(s_flat.float(), y_flat, reduction="mean")
            loss_mse = F.mse_loss(h_corr.reshape(-1, 5120).float(), y_tr_63.float())
            loss_total = loss_kl + 0.1 * loss_ce + 0.05 * loss_mse

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            if step in [1, 2, 3, 5, 10, 15, 20, 25, 30]:
                model.eval()
                with torch.no_grad():
                    h_eval = model(h_63_raw)
                    l_tr = F.linear(norm_mod(h_eval[:num_train]), lm_head_w)
                    l_te = F.linear(norm_mod(h_eval[num_train:]), lm_head_w)
                ev_tr = compute_metrics(l_tr, train_targets)
                ev_te = compute_metrics(l_te, test_targets, teacher_logits=logits_prof_test)
                delta_ppl = ev_te["ppl"] - m_te_base["ppl"]

                rec = {
                    "step": step,
                    "loss": loss_total.item(),
                    "train_nll": ev_tr["nll"],
                    "train_ppl": ev_tr["ppl"],
                    "test_nll": ev_te["nll"],
                    "test_ppl": ev_te["ppl"],
                    "test_top1": ev_te["top1"],
                    "test_cos": ev_te["cos_sim"],
                    "test_kl": ev_te["kl_div"],
                    "delta_ppl": delta_ppl
                }
                history.append(rec)
                print(f"  Step {step:02d} | Loss: {loss_total.item():.4f} | Train PPL: {ev_tr['ppl']:6.2f} | Test PPL: {ev_te['ppl']:6.2f} (Delta: {delta_ppl:+6.2f}) | Top-1: {ev_te['top1']:5.2f}% | KL: {ev_te['kl_div']:.4f}")

                if ev_te["ppl"] < best_overall_ppl:
                    best_overall_ppl = ev_te["ppl"]
                    best_config_name = f"{cfg_name} (Step {step})"

        all_experiments[cfg_name] = {
            "config": {k: cfg_item[k] for k in ["name", "r_c", "r_s", "r_tot"]},
            "param_count": param_count,
            "history": history,
            "final_metrics": history[-1]
        }

    print("\n" + "=" * 115)
    print(f"RESUMO COMPARATIVO TERMINAL: MELHOR RESULTADO GLOBAL = {best_config_name} (PPL = {best_overall_ppl:.2f})")
    print("=" * 115)
    print(f"{'Configuracao':<30} | {'Canais':<8} | {'Parametros':<12} | {'Test PPL':<10} | {'Delta vs SVD-256':<18} | {'Top-1':<8} | {'KL':<8}")
    print("-" * 115)
    print(f"{'Baseline SVD-256 (r=256)':<30} | {'256':<8} | {'2,621,440':<12} | {m_te_base['ppl']:<10.2f} | {'0.00 (Ref)':<18} | {m_te_base['top1']:<7.2f}% | {m_te_base['kl_div']:<8.4f}")
    print(f"{'L63 Monolitico (Doc 13)':<30} | {'256':<8} | {'2,621,440':<12} | {'179.94':<10} | {'-12.74 (Recorde)':<18} | {'23.21%':<8} | {'2.9814':<8}")
    print("-" * 115)
    for cfg_name, exp in all_experiments.items():
        m = exp["final_metrics"]
        d_ppl = m["test_ppl"] - m_te_base["ppl"]
        c_tot = exp["config"]["r_tot"]
        p_cnt = exp["param_count"]
        print(f"{cfg_name:<30} | {c_tot:<8} | {p_cnt:<12,} | {m['test_ppl']:<10.2f} | {d_ppl:<+18.2f} | {m['test_top1']:<7.2f}% | {m['test_kl']:<8.4f}")
    print("=" * 115)

    # 4. Salvar Checkpoint e JSON
    out_json = os.path.join(repo_root, "experiments", "hybrid_terminal_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "baselines": {
                "raw_atlas": m_raw_te,
                "svd256_uniform": m_te_base,
                "doc13_monolithic": {"ppl": 179.94, "nll": 5.1926, "top1": 23.21, "kl": 2.9814}
            },
            "experiments": all_experiments
        }, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {out_json}")

    # 5. Gerar Grafico Comparativo
    plt.figure(figsize=(11, 6))
    steps_eval = [h["step"] for h in list(all_experiments.values())[0]["history"]]
    plt.axhline(y=m_te_base["ppl"], color="red", linestyle="--", linewidth=2, label="Baseline SVD-256 Uniforme (PPL = 192.68)")
    plt.axhline(y=179.94, color="green", linestyle=":", linewidth=2, label="L63 Otimizado Monolítico r=256 (Doc 13, PPL = 179.94)")

    colors = ["tab:blue", "tab:orange", "tab:purple"]
    markers = ["o", "s", "^"]
    for idx, (cfg_name, exp) in enumerate(all_experiments.items()):
        ppls = [h["test_ppl"] for h in exp["history"]]
        plt.plot(steps_eval, ppls, label=f"{cfg_name} (r_tot={exp['config']['r_tot']})", color=colors[idx], marker=markers[idx], linewidth=2.2)

    plt.title("Otimização Terminal L63: Refinador Híbrido com Bases Congeladas vs SVD-256", fontsize=13, fontweight="bold")
    plt.xlabel("Passo de Treinamento (AdamW)", fontsize=11)
    plt.ylabel("Perplexidade no WikiText-2 Test (PPL)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plot_out = os.path.join(repo_root, "hybrid_terminal_ppl_curve.png")
    plt.savefig(plot_out, dpi=300)
    plt.close()
    print(f"[OK] Grafico salvo em: {plot_out}")

if __name__ == "__main__":
    main()
