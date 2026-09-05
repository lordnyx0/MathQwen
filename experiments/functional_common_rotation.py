# -*- coding: utf-8 -*-
"""
EXPERIMENTOS C, D e E: OTIMIZACAO DE ROTACAO FUNCIONAL E DEFORMACAO GRASSMANNIANA EM L63
- Experimento C: Otimização de rotação ortogonal Q in SO(32) via álgebra de Lie so(32): Q(A) = Q_0 exp(A - A^T).
- Experimento D: Comparação D1 (Q = I fixo), D2 (Q inicializado por G_c) e D3 (Q aleatório perto de I).
- Experimento E: Deformação geodésica no Grassmanniano U(theta) = Exp_{U_c}(Delta) e curva PPL vs d_Gr.
- Avalia steps em {0, 1, 2, 3, 5, 10} com early-stopping e registra PPL, Top-1 e KL.
- Salva gráfico functional_rotation_curve.png.
"""
import os
import sys
import time
import json
import math
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

class FunctionalRotatedStabilizer(nn.Module):
    def __init__(self, U_c: torch.Tensor, Q_init: torch.Tensor = None, learn_rotation: bool = True, alpha: float = 1.0):
        super().__init__()
        r = U_c.shape[1]
        self.r = r
        self.register_buffer("U_c", U_c.clone().to(dtype=torch.bfloat16))

        if Q_init is None:
            Q_init = torch.eye(r, device=U_c.device, dtype=torch.float32)
        else:
            Q_init = Q_init.to(device=U_c.device, dtype=torch.float32)

        self.register_buffer("Q_base", Q_init)
        self.learn_rotation = learn_rotation

        if learn_rotation:
            self.A = nn.Parameter(torch.zeros(r, r, device=U_c.device, dtype=torch.float32))
        else:
            self.A = None

        self.W = nn.Parameter(torch.zeros(r, 5120, device=U_c.device, dtype=torch.bfloat16))
        self.alpha = nn.Parameter(torch.tensor([alpha], device=U_c.device, dtype=torch.float32))

    def get_Q(self) -> torch.Tensor:
        if not self.learn_rotation:
            return self.Q_base
        Omega = 0.5 * (self.A - self.A.t())
        Q_rot = torch.linalg.matrix_exp(Omega)
        return torch.matmul(self.Q_base, Q_rot)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        Q = self.get_Q().to(dtype=torch.bfloat16)
        U_f = torch.matmul(self.U_c, Q) # (5120, r)
        corr = torch.matmul(torch.matmul(h, U_f), self.W)
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
    print("EXPERIMENTOS C, D e E: OTIMIZACAO DE ROTACAO FUNCIONAL E DEFORMACAO GRASSMANNIANA EM L63")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Baseline Canônico SVD-256 Uniforme: PPL = 192.68")
    print("Pure Universal U_c(32) (Doc 15):     PPL = 186.63")
    print("Recorde L63 Monolitico (Doc 13):      PPL = 179.94")

    # 1. Carregar Bases U_c(32) e Q_sorted funcional do Experimento A
    bases_path = os.path.join(repo_root, "checkpoints", "common_specific_bases_l63.pt")
    saved_bases = torch.load(bases_path, map_location=device)
    U_c_32 = saved_bases["U_c"][32].to(device=device, dtype=torch.bfloat16)

    func_ckpt_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    if os.path.exists(func_ckpt_path):
        saved_func = torch.load(func_ckpt_path, map_location=device)
        Q_func = saved_func["Q"].float().to(device)
    else:
        print("[AVISO] Checkpoint funcional Uc_functional_r32.pt nao encontrado. Usando I.")
        Q_func = torch.eye(32, device=device, dtype=torch.float32)

    # 2. Carregar Targets e Componentes de Referência
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
    with torch.no_grad():
        l_te_raw = F.linear(norm_mod(h_63_raw[num_train:]), lm_head_w)
        m_raw_te = compute_metrics(l_te_raw, test_targets, teacher_logits=logits_prof_test)

    print("-" * 85)
    print(f"Atlas Raw L63 (sem refiner)  : NLL={m_raw_te['nll']:.4f} | PPL={m_raw_te['ppl']:.2f} | Top-1={m_raw_te['top1']:.2f}% | KL={m_raw_te['kl_div']:.4f}")
    print(f"Baseline SVD-256 L63 (r=256) : PPL=192.68 | Top-1=22.72% | KL=3.1317")
    print(f"Pure Universal U_c(32)       : PPL=186.63 | Top-1=23.21% | KL=3.1067")
    print("-" * 85)

    # 3. Definir as Condições D1, D2, D3
    conditions = [
        {
            "id": "D1",
            "name": "D1: Q = I (Fixo, sem rotação)",
            "Q_init": torch.eye(32, device=device, dtype=torch.float32),
            "learn_rotation": False,
            "lr": 1e-4
        },
        {
            "id": "D2",
            "name": "D2: Q = Q_func (Inicializado por G_c)",
            "Q_init": Q_func.clone(),
            "learn_rotation": False,
            "lr": 1e-4
        },
        {
            "id": "D2_trainable",
            "name": "D2*: Q_func + Rotação Aprendida em SO(32)",
            "Q_init": Q_func.clone(),
            "learn_rotation": True,
            "lr": 1e-4
        },
        {
            "id": "D3",
            "name": "D3: Q Aleatório Próximo de I",
            "Q_init": torch.eye(32, device=device, dtype=torch.float32),
            "learn_rotation": True,
            "lr": 1e-4,
            "random_init": True
        }
    ]

    eval_steps = [0, 1, 2, 3, 5, 10]
    exp_results = {}
    best_ppl_global = 9999.0
    best_cond_name = None

    for cond in conditions:
        c_id = cond["id"]
        c_name = cond["name"]
        print(f"\n=========================================================================================")
        print(f"EXECUTANDO CONDICAO {c_name}")
        print(f"=========================================================================================")

        model = FunctionalRotatedStabilizer(
            U_c_32,
            Q_init=cond["Q_init"],
            learn_rotation=cond["learn_rotation"]
        ).to(device=device)

        if cond.get("random_init", False) and model.A is not None:
            nn.init.normal_(model.A, mean=0.0, std=0.02)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_cnt = sum(p.numel() for p in trainable_params)
        print(f"  -> Parâmetros treináveis: {param_cnt:,} ({param_cnt / (2*5120*256) * 100:.2f}% do SVD-256)")

        optimizer = torch.optim.AdamW(trainable_params, lr=cond["lr"], weight_decay=1e-3)
        history = []

        # Step 0 (Avaliação inicial antes do treino)
        model.eval()
        with torch.no_grad():
            h_eval0 = model(h_63_raw)
            l_tr0 = F.linear(norm_mod(h_eval0[:num_train]), lm_head_w)
            l_te0 = F.linear(norm_mod(h_eval0[num_train:]), lm_head_w)
        ev_tr0 = compute_metrics(l_tr0, train_targets)
        ev_te0 = compute_metrics(l_te0, test_targets, teacher_logits=logits_prof_test)
        print(f"  Step 00 | Loss: Init   | Train PPL: {ev_tr0['ppl']:6.2f} | Test PPL: {ev_te0['ppl']:6.2f} | Top-1: {ev_te0['top1']:5.2f}% | KL: {ev_te0['kl_div']:.4f}")
        history.append({
            "step": 0, "loss": 0.0, "train_ppl": ev_tr0["ppl"],
            "test_ppl": ev_te0["ppl"], "test_top1": ev_te0["top1"], "test_kl": ev_te0["kl_div"]
        })

        for step in range(1, 11):
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

            if step in eval_steps:
                model.eval()
                with torch.no_grad():
                    h_eval = model(h_63_raw)
                    l_tr = F.linear(norm_mod(h_eval[:num_train]), lm_head_w)
                    l_te = F.linear(norm_mod(h_eval[num_train:]), lm_head_w)
                ev_tr = compute_metrics(l_tr, train_targets)
                ev_te = compute_metrics(l_te, test_targets, teacher_logits=logits_prof_test)
                delta_ppl = ev_te["ppl"] - 192.68

                rec = {
                    "step": step,
                    "loss": loss_total.item(),
                    "train_ppl": ev_tr["ppl"],
                    "test_ppl": ev_te["ppl"],
                    "test_top1": ev_te["top1"],
                    "test_cos": ev_te["cos_sim"],
                    "test_kl": ev_te["kl_div"],
                    "delta_ppl": delta_ppl
                }
                history.append(rec)
                print(f"  Step {step:02d} | Loss: {loss_total.item():.4f} | Train PPL: {ev_tr['ppl']:6.2f} | Test PPL: {ev_te['ppl']:6.2f} (Delta: {delta_ppl:+6.2f}) | Top-1: {ev_te['top1']:5.2f}% | KL: {ev_te['kl_div']:.4f}")

                if ev_te["ppl"] < best_ppl_global:
                    best_ppl_global = ev_te["ppl"]
                    best_cond_name = f"{c_id} (Step {step})"

        exp_results[c_id] = {
            "name": c_name,
            "param_cnt": param_cnt,
            "history": history,
            "best_ppl": min(h["test_ppl"] for h in history if h["step"] > 0)
        }

    # 4. Experimento E: Deformação Geodésica no Grassmanniano
    print("\n=========================================================================================")
    print("EXPERIMENTO E: DEFORMACAO GEODESICA NO GRASSMANNIANO U(theta) = Exp_{U_c}(Delta)")
    print("=========================================================================================")
    # Construir direção ortogonal Delta: Delta^T U_c = 0, Delta^T Delta = I_32
    torch.manual_seed(42)
    R_rand = torch.randn(5120, 32, device=device, dtype=torch.float32)
    # Projetar para fora de U_c: Delta = (I - U_c U_c^T) R
    R_orth = R_rand - torch.matmul(U_c_32.float(), torch.matmul(U_c_32.float().t(), R_rand))
    Delta, _ = torch.linalg.qr(R_orth) # (5120, 32)

    angles_deg_eval = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0]
    grassmann_curve = []

    # Usar o W_f otimizado da melhor condição D para avaliar a sensibilidade geodésica
    W_eval = model.W.data.clone() # (32, 5120)

    for theta_deg in angles_deg_eval:
        theta_rad = theta_deg * (math.pi / 180.0)
        # Geodésica no Stiefel/Grassmanniano: U(theta) = U_c cos(theta) + Delta sin(theta)
        U_theta = (U_c_32.float() * math.cos(theta_rad) + Delta * math.sin(theta_rad)).to(dtype=torch.bfloat16)
        
        with torch.no_grad():
            corr = torch.matmul(torch.matmul(h_63_raw[num_train:], U_theta), W_eval)
            h_def = h_63_raw[num_train:] + corr
            l_def = F.linear(norm_mod(h_def), lm_head_w)
            m_def = compute_metrics(l_def, test_targets, teacher_logits=logits_prof_test)

        grassmann_curve.append({
            "theta_deg": theta_deg,
            "ppl": m_def["ppl"],
            "top1": m_def["top1"],
            "kl": m_def["kl_div"]
        })
        print(f"  Angulo Geodesico theta: {theta_deg:5.1f} deg | Test PPL: {m_def['ppl']:6.2f} | Top-1: {m_def['top1']:5.2f}% | KL: {m_def['kl_div']:.4f}")

    # 5. Salvar Gráfico: functional_rotation_curve.png
    fig, (ax_rot, ax_gr) = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle("Experimentos C, D e E: Otimização de Rotação em SO(32) e Deformação no Grassmanniano (L63)", fontsize=14, fontweight="bold")

    # Plot 1: Curvas de PPL para D1, D2, D2*, D3 ao longo dos passos
    styles = {
        "D1": ("tab:blue", "--", "o"),
        "D2": ("tab:green", "-", "s"),
        "D2_trainable": ("tab:red", "-", "^"),
        "D3": ("tab:purple", ":", "d")
    }
    for c_id, res in exp_results.items():
        steps_plot = [h["step"] for h in res["history"]]
        ppls_plot = [h["test_ppl"] for h in res["history"]]
        color, ls, marker = styles.get(c_id, ("black", "-", "."))
        ax_rot.plot(steps_plot, ppls_plot, color=color, linestyle=ls, marker=marker, lw=2, label=f"{c_id} (min: {res['best_ppl']:.2f})")

    ax_rot.axhline(192.68, color="gray", linestyle="--", label="Baseline SVD-256 (192.68)")
    ax_rot.axhline(179.94, color="gold", linestyle=":", label="Recorde L63 Mono (179.94)")
    ax_rot.set_xlabel("Passos de Otimização Terminal", fontweight="bold")
    ax_rot.set_ylabel("Test Perplexity (PPL)", fontweight="bold")
    ax_rot.set_title("Evolução da PPL: Comparação D1 vs D2 vs D3", fontweight="bold")
    ax_rot.grid(True, alpha=0.3)
    ax_rot.legend()

    # Plot 2: Curva PPL vs Distância Grassmanniana theta
    t_vals = [gc["theta_deg"] for gc in grassmann_curve]
    p_vals = [gc["ppl"] for gc in grassmann_curve]
    ax_gr.plot(t_vals, p_vals, marker="o", color="tab:red", lw=2.5, label="PPL(theta)")
    ax_gr.set_xlabel("Deformação Geodésica theta no Grassmanniano (graus)", fontweight="bold")
    ax_gr.set_ylabel("Test Perplexity (PPL)", fontweight="bold")
    ax_gr.set_title("Experimento E: Sensibilidade à Deformação fora de U_c", fontweight="bold")
    ax_gr.grid(True, alpha=0.3)
    ax_gr.legend()

    plt.tight_layout()
    rot_chart_path = os.path.join(repo_root, "experiments", "functional_rotation_curve.png")
    fig.savefig(rot_chart_path, dpi=300)
    fig.savefig(os.path.join(repo_root, "functional_rotation_curve.png"), dpi=300)
    plt.close(fig)
    print(f"\n[OK] Gráfico salvo em: {rot_chart_path} e raiz do repositório.")

    # 6. Salvar Resultados JSON
    rot_json_path = os.path.join(repo_root, "experiments", "functional_rotation_results.json")
    with open(rot_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparisons": exp_results,
            "grassmann_curve": grassmann_curve,
            "best_ppl_global": best_ppl_global,
            "best_cond_name": best_cond_name
        }, f, indent=2)
    print(f"[OK] Resultados salvos em: {rot_json_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
