# -*- coding: utf-8 -*-
"""
Common + Specific Error Manifold: Construcao do Refinador Hibrido e Testes de Generalizacao
Implementa:
h' = h + (h U_c) W_c + (h V_e) W_e
Testa:
1. Variante 1: Puro Universal C_common (Treinado em A, avaliado em B e C)
2. Variante 2: Comum + Especifico C_e = C_common + C_specific^(e)
3. Teste sem rotulo de dominio (E_pool sem supervisao)
4. Avaliacao de NLL, PPL, Top-1 e KL no WikiText-2 Test Set e retencao de erro funcional MSE.
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
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

def fit_ridge(Z: torch.Tensor, E: torch.Tensor, lambda_reg: float = 1e-3):
    """
    Regressao Ridge fechada: W = (Z^T Z + lambda I)^{-1} Z^T E
    Z: (N, r), E: (N, 5120) -> W: (r, 5120)
    """
    r = Z.shape[1]
    ZTZ = torch.matmul(Z.t(), Z)
    reg_val = lambda_reg * (torch.trace(ZTZ) / float(r))
    reg_I = reg_val * torch.eye(r, device=Z.device, dtype=torch.float32)
    L = torch.linalg.cholesky(ZTZ + reg_I)
    W = torch.cholesky_solve(torch.matmul(Z.t(), E), L)
    return W

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print("COMMON + SPECIFIC ERROR MANIFOLD - REFINADOR HIBRIDO E TESTES DE GENERALIZACAO")
    print("=" * 105)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar Bases U_c e V_specific
    bases_path = os.path.join(repo_root, "checkpoints", "common_specific_bases_l63.pt")
    if not os.path.exists(bases_path):
        raise FileNotFoundError(f"Checkpoint {bases_path} nao encontrado!")

    saved_bases = torch.load(bases_path, map_location=device)
    U_c_dict = {k: v.to(device) for k, v in saved_bases["U_c"].items()}
    V_spec_dict = {k: {d: v.to(device) for d, v in d_dict.items()} for k, d_dict in saved_bases["V_specific"].items()}

    # 2. Carregar Estados X e Erros E
    err_path = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")
    raw_errors = torch.load(err_path, map_location=device)
    domains = ["WikiText", "HF_Code", "GSM8K"]
    E_dict = {d: raw_errors[d].to(device).float() for d in domains}

    h_wiki_path = os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt")
    h_hf_path = os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt")
    h_gsm8k_path = os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt")

    X_dict = {
        "WikiText": torch.load(h_wiki_path, map_location=device).reshape(-1, 5120).to(device, dtype=torch.bfloat16),
        "HF_Code": torch.load(h_hf_path, map_location=device)["h_in_63"].to(device).reshape(-1, 5120).to(torch.bfloat16),
        "GSM8K": torch.load(h_gsm8k_path, map_location=device)["h_in_63"].to(device).reshape(-1, 5120).to(torch.bfloat16)
    }

    norm_orig_sq = {d: (E_dict[d] ** 2).sum().item() for d in domains}

    # 3. Teste 1: Variante 1 - Universal Puro (C_common) Transferindo entre Dominios
    print("\n[Passo 1/4] Avaliando Variante 1: Universal Puro (C_common) Out-of-Domain...")
    universal_records = []

    for r_c in [8, 16, 32, 64]:
        U_c = U_c_dict[r_c]
        for d_fit in domains:
            X_fit = X_dict[d_fit].float()
            E_fit = E_dict[d_fit]
            Z_fit = torch.matmul(X_fit, U_c)
            W_c = fit_ridge(Z_fit, E_fit)

            # Testar em todos os dominios
            ret_test = {}
            for d_test in domains:
                X_test = X_dict[d_test].float()
                E_test = E_dict[d_test]
                Z_test = torch.matmul(X_test, U_c)
                pred_err = torch.matmul(Z_test, W_c)
                res = E_test - pred_err
                res_sq = (res ** 2).sum().item()
                captured_pct = max(0.0, 1.0 - res_sq / norm_orig_sq[d_test]) * 100.0
                ret_test[d_test] = captured_pct

            universal_records.append({
                "r_c": r_c,
                "fit_domain": d_fit,
                "retentions": ret_test,
                "worst_retention": min(ret_test.values()),
                "mean_retention": float(np.mean(list(ret_test.values())))
            })
            out_str = " | ".join([f"{d}: {ret_test[d]:5.2f}%" for d in domains])
            print(f"  r_c={r_c:<2} | Treino: {d_fit:<10} | [{out_str}] | Pior Caso: {min(ret_test.values()):5.2f}%")

    # 4. Teste 2: Variante 2 - Hibrido Comum + Especifico (C_common + C_specific)
    print("\n[Passo 2/4] Avaliando Variante 2: Hibrido Comum + Especifico (C_common + C_specific)...")
    hybrid_records = []

    # O nucleo comum U_c e treinado em Pool (A+B+C) ou em WikiText
    X_pool = torch.cat([X_dict[d].float() for d in domains], dim=0)
    E_pool = torch.cat([E_dict[d] for d in domains], dim=0)

    for r_c in [16, 32, 64]:
        U_c = U_c_dict[r_c]
        # Ajustar W_c comum no Pool
        Z_pool_c = torch.matmul(X_pool, U_c)
        W_c_pool = fit_ridge(Z_pool_c, E_pool)

        for r_s in [8, 16, 32]:
            rec_hyb = {"r_c": r_c, "r_s": r_s, "total_rank": r_c + r_s, "retentions": {}}
            for d in domains:
                X_d = X_dict[d].float()
                E_d = E_dict[d]
                V_s = V_spec_dict[r_c][d][:, :r_s]

                # Modelo Hibrido Conjunto: Z = [X @ U_c, X @ V_s]
                Z_c = torch.matmul(X_d, U_c)
                Z_s = torch.matmul(X_d, V_s)
                Z_joint = torch.cat([Z_c, Z_s], dim=1)
                W_joint = fit_ridge(Z_joint, E_d)

                pred_joint = torch.matmul(Z_joint, W_joint)
                res_joint = E_d - pred_joint
                res_sq = (res_joint ** 2).sum().item()
                cap_pct = max(0.0, 1.0 - res_sq / norm_orig_sq[d]) * 100.0
                rec_hyb["retentions"][d] = cap_pct

            rec_hyb["worst_retention"] = min(rec_hyb["retentions"].values())
            rec_hyb["mean_retention"] = float(np.mean(list(rec_hyb["retentions"].values())))
            hybrid_records.append(rec_hyb)
            out_str = " | ".join([f"{d}: {rec_hyb['retentions'][d]:5.2f}%" for d in domains])
            print(f"  (r_c={r_c:<2}, r_s={r_s:<2}, r_tot={r_c+r_s:<2}) | [{out_str}] | Pior: {rec_hyb['worst_retention']:5.2f}% | Media: {rec_hyb['mean_retention']:5.2f}%")

    # 5. Teste 3: Descoberta Sem Rotulo de Dominio (Unsupervised Shared Core)
    print("\n[Passo 3/4] Executando Teste de Compartilhamento Sem Rotulo de Dominio...")
    # Embaralhar E_pool aleatoriamente
    perm = torch.randperm(E_pool.shape[0], device=device)
    E_shuffled = E_pool[perm]
    Sigma_unsupervised = torch.matmul(E_shuffled.t(), E_shuffled) / float(E_shuffled.shape[0])
    eigvals_un, eigvecs_un = torch.linalg.eigh(Sigma_unsupervised)
    eigvecs_un = torch.flip(eigvecs_un, dims=[1]) # Ordem decrescente

    unlabelled_records = {}
    for r_c in [8, 16, 32, 64]:
        U_un = eigvecs_un[:, :r_c]
        ret_un = {}
        for d in domains:
            proj_sq = (torch.matmul(E_dict[d], U_un) ** 2).sum().item()
            ret_un[d] = (proj_sq / norm_orig_sq[d]) * 100.0
        worst_un = min(ret_un.values())
        mean_un = float(np.mean(list(ret_un.values())))
        unlabelled_records[str(r_c)] = {
            "domain_retentions": ret_un,
            "worst_retention": worst_un,
            "mean_retention": mean_un
        }
        print(f"  r_c={r_c:<2} Sem Rotulo: Media={mean_un:5.2f}% | Pior={worst_un:5.2f}% | Detalhes: { {k: round(v, 2) for k, v in ret_un.items()} }")

    # 6. Teste 4: Avaliacao de PPL/NLL/KL/Top1 no WikiText-2 Test Set
    print("\n[Passo 4/4] Avaliando PPL/NLL/Top-1/KL no WikiText-2 Test Set...")
    snapshot_dir = get_snapshot_dir()
    comp = load_qwen_reference_components(device=device, seq_len=64, num_seqs=16)
    lm_head_w = comp["lm_head_w"]
    norm_mod = comp["norm_mod"]

    p_data = torch.load("checkpoints/prof_targets_cache.pt", map_location=device)
    logits_prof_test = p_data["logits_prof_test"].to(device)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tok(raw_text, return_tensors="pt")["input_ids"][0]
    test_ids = tokens_all[16*64:32*64].view(16, 64).to(device)
    test_targets = test_ids[:, 1:].contiguous().reshape(-1)

    # Estado de entrada L63 test
    h63_wiki = torch.load("checkpoints/h_stud_63_cache.pt", map_location=device).to(device)
    h_in_test = h63_wiki[16:]

    bases_raw = torch.load("checkpoints/atlas_bases.pt", map_location=device)
    chart15_bases = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in bases_raw[15].items()}

    l63_atlas = load_layer_module(63, comp["cfg"], device, snapshot_dir=snapshot_dir)
    import safetensors.torch as st
    lp = os.path.join(snapshot_dir, "layers-63.safetensors")
    with st.safe_open(lp, framework="pt") as f:
        mix_w = dequant(f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight"), f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        gate_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        up_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        down_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        cw = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": True}
    p_w = project_layer_weights(cw, chart15_bases)
    l63_atlas.self_attn.o_proj.weight.data.copy_(p_w["mix"])
    l63_atlas.mlp.gate_proj.weight.data.copy_(p_w["gate"])
    l63_atlas.mlp.up_proj.weight.data.copy_(p_w["up"])
    l63_atlas.mlp.down_proj.weight.data.copy_(p_w["down"])

    pos_ids = torch.arange(64, device=device).view(1, 1, -1).expand(3, 16, -1)
    pos_emb = comp["rotary"](h_in_test, pos_ids)
    with torch.no_grad():
        out_s = l63_atlas(h_in_test, position_embeddings=pos_emb)
        h_63_raw_test = (out_s[0] if isinstance(out_s, tuple) else out_s).to(torch.bfloat16)

    del l63_atlas
    torch.cuda.empty_cache()

    def eval_h(h_state):
        with torch.no_grad():
            x_norm = norm_mod(h_state)
            logits = F.linear(x_norm, lm_head_w)
            vocab_size = logits.shape[-1]
            l_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
            nll = F.cross_entropy(l_pred, test_targets, reduction="mean").item()
            ppl = math.exp(nll)
            top1 = (l_pred.argmax(dim=-1) == test_targets).float().mean().item() * 100.0
            t_pred = logits_prof_test[:, :-1, :].reshape(-1, vocab_size).float()
            kl_div = F.kl_div(F.log_softmax(l_pred, dim=-1), F.softmax(t_pred, dim=-1), reduction="batchmean").item()
        return {"nll": nll, "ppl": ppl, "top1": top1, "kl": kl_div}

    m_raw = eval_h(h_63_raw_test)
    stabs_raw = torch.load("checkpoints/svd256_baseline_stabilizers.pt", map_location=device)
    stab_base = LinearResidualStabilizer(stabs_raw[63]["W_down"], stabs_raw[63]["W_up"], alpha=stabs_raw[63].get("alpha", 1.0)).to(device, dtype=torch.bfloat16)
    m_base = eval_h(stab_base(h_63_raw_test))

    print(f"Baseline Raw Atlas L63 : NLL={m_raw['nll']:.4f} | PPL={m_raw['ppl']:.2f} | Top-1={m_raw['top1']:.2f}% | KL={m_raw['kl']:.4f}")
    print(f"Baseline SVD-256 L63    : NLL={m_base['nll']:.4f} | PPL={m_base['ppl']:.2f} | Top-1={m_base['top1']:.2f}% | KL={m_base['kl']:.4f}")

    # Salvar todos os registros
    final_results = {
        "universal_variant1": universal_records,
        "hybrid_variant2": hybrid_records,
        "unlabelled_test": unlabelled_records,
        "baselines": {"raw": m_raw, "svd256": m_base}
    }

    out_file = os.path.join(repo_root, "experiments", "common_specific_refiner.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {out_file}")

if __name__ == "__main__":
    main()
