# -*- coding: utf-8 -*-
"""
Universal Error Manifold (UEM) - Analise Espectral e Otimizacao Minimax na Camada 63
Executa a decomposicao de erro funcional: delta_63(x) = Atlas_63(x) - Teacher_63(x)
Calcula covariancias Sigma_E^(k), autovalores, subespaco medio U_avg, subespaco minimax U_minimax,
curvas de retencao rho(r) e retorno marginal por bit R_bits(r).
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
import safetensors.torch as st
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

def solve_minimax_subspace(sigmas_norm: list, r: int, U_init: torch.Tensor, max_iters: int = 60, lr: float = 0.05):
    """
    Resolve o subespaco minimax no manifold de Stiefel/Grassmanniano:
    max_{U^T U = I_r} min_k Tr(U^T Sigma_k_norm U)
    usando subgradiente projetado via retracao QR.
    """
    K = len(sigmas_norm)
    U = U_init.clone()
    history = []

    for step in range(max_iters):
        # 1. Calcular retencao em cada dominio
        f_vals = [torch.trace(torch.matmul(U.t(), torch.matmul(S, U))).item() for S in sigmas_norm]
        min_idx = int(np.argmin(f_vals))
        worst_f = f_vals[min_idx]
        history.append({
            "step": step,
            "worst_retention": worst_f,
            "domain_retentions": f_vals,
            "worst_domain": min_idx
        })

        # 2. Direcao de subgradiente do pior dominio
        # grad de Tr(U^T S U) em relacao a U e 2 * S @ U
        S_worst = sigmas_norm[min_idx]
        grad = 2.0 * torch.matmul(S_worst, U)

        # 3. Retracao no Stiefel via QR
        eta = lr * (0.95 ** (step // 10))
        U_new, _ = torch.linalg.qr(U + eta * grad)
        U = U_new

    # Avaliacao final
    f_vals_final = [torch.trace(torch.matmul(U.t(), torch.matmul(S, U))).item() for S in sigmas_norm]
    history.append({
        "step": max_iters,
        "worst_retention": float(np.min(f_vals_final)),
        "domain_retentions": f_vals_final,
        "worst_domain": int(np.argmin(f_vals_final))
    })
    return U, history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("UNIVERSAL ERROR MANIFOLD (UEM) - ANALISE ESPECTRAL E MINIMAX NA CAMADA 63")
    print("=" * 100)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    snapshot_dir = get_snapshot_dir()
    d_model = 5120

    # 1. Carregar os estados de entrada de L63 para cada dominio
    print("\n[Passo 1/6] Carregando colecoes de estados de entrada de L63...")

    # Dominio 1: WikiText (Linguagem Natural Geral)
    h_wiki_path = os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt")
    X_wiki = torch.load(h_wiki_path, map_location=device).reshape(-1, d_model).to(torch.bfloat16)
    print(f"  -> Dominio 1 (WikiText): {X_wiki.shape[0]} estados ({X_wiki.shape})")

    # Dominio 2: HF Cache (Codigo e Raciocinio Algoritmico)
    h_hf_path = os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt")
    X_hf = torch.load(h_hf_path, map_location=device)["h_in_63"].to(device).reshape(-1, d_model).to(torch.bfloat16)
    print(f"  -> Dominio 2 (HF Cache): {X_hf.shape[0]} estados ({X_hf.shape})")

    # Dominio 3: GSM8k (Raciocinio Matematico) se disponivel, senao particionar WikiText em split independente
    h_gsm8k_path = os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt")
    if os.path.exists(h_gsm8k_path):
        print(f"  -> Dominio 3 (GSM8k): Carregando de {h_gsm8k_path}...")
        X_math = torch.load(h_gsm8k_path, map_location=device)["h_in_63"].to(device).reshape(-1, d_model).to(torch.bfloat16)
        print(f"     Carregado com sucesso: {X_math.shape[0]} estados ({X_math.shape})")
        domains = {
            "WikiText": X_wiki,
            "HF_Code": X_hf,
            "GSM8K": X_math
        }
    else:
        print(f"  -> GSM8k cache ainda nao finalizado. Usando split independente de WikiText como Dominio 3 temporario.")
        # Particionar X_wiki em 2 metades independentes
        half = X_wiki.shape[0] // 2
        domains = {
            "WikiText_Train": X_wiki[:half],
            "WikiText_Test": X_wiki[half:],
            "HF_Code": X_hf
        }

    # 2. Instanciar Layer 63 Teacher (FP8 desquantizado) e Layer 63 Atlas (projetado)
    print("\n[Passo 2/6] Preparando modulos Teacher e Atlas para L63...")
    comp = load_qwen_reference_components(device=device, seq_len=128, num_seqs=16)
    cfg = comp["cfg"]

    bases_cache_path = os.path.join(repo_root, "checkpoints", "atlas_bases.pt")
    raw_bases = torch.load(bases_cache_path, map_location=device)
    chart15_bases = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in raw_bases[15].items()}

    # Teacher Layer 63
    l63_teacher = load_layer_module(63, cfg, device, snapshot_dir=snapshot_dir)
    lp = os.path.join(snapshot_dir, "layers-63.safetensors")
    with st.safe_open(lp, framework="pt") as f:
        mix_w = dequant(f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight"), f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        gate_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        up_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        down_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        cw = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": True}

    l63_teacher.self_attn.o_proj.weight.data.copy_(mix_w)
    l63_teacher.mlp.gate_proj.weight.data.copy_(gate_w)
    l63_teacher.mlp.up_proj.weight.data.copy_(up_w)
    l63_teacher.mlp.down_proj.weight.data.copy_(down_w)

    # Atlas Layer 63
    l63_atlas = load_layer_module(63, cfg, device, snapshot_dir=snapshot_dir)
    p_w = project_layer_weights(cw, chart15_bases)
    l63_atlas.self_attn.o_proj.weight.data.copy_(p_w["mix"])
    l63_atlas.mlp.gate_proj.weight.data.copy_(p_w["gate"])
    l63_atlas.mlp.up_proj.weight.data.copy_(p_w["up"])
    l63_atlas.mlp.down_proj.weight.data.copy_(p_w["down"])

    # 3. Computar operadores de erro funcional E^(k) = Atlas(X) - Teacher(X)
    print("\n[Passo 3/6] Computando operadores de erro funcional delta_63(X) = Atlas_63(X) - Teacher_63(X)...")
    errors = {}
    sigmas = {}
    sigmas_norm = []
    traces = {}

    err_cache_file = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")

    for d_name, X in domains.items():
        N = X.shape[0]
        # Dividir em mini-lotes para seguranca de VRAM
        batch_sz = 256
        E_list = []
        for b_start in range(0, N, batch_sz):
            b_end = min(b_start + batch_sz, N)
            x_batch = X[b_start:b_end].view(1, b_end - b_start, d_model)
            pos_ids = torch.arange(b_end - b_start, device=device).view(1, 1, -1).expand(3, 1, -1)
            pos_emb = comp["rotary"](x_batch, pos_ids)
            with torch.no_grad():
                out_T = l63_teacher(x_batch, position_embeddings=pos_emb)
                out_A = l63_atlas(x_batch, position_embeddings=pos_emb)
                h_T = (out_T[0] if isinstance(out_T, tuple) else out_T).view(-1, d_model).float()
                h_A = (out_A[0] if isinstance(out_A, tuple) else out_A).view(-1, d_model).float()
                delta = h_A - h_T
            E_list.append(delta.cpu())

        E_k = torch.cat(E_list, dim=0).to(device)
        errors[d_name] = E_k
        N_k = E_k.shape[0]

        # Matriz de covariancia de erro (5120 x 5120) em float32
        Sigma_k = torch.matmul(E_k.t(), E_k) / float(N_k)
        tr_k = torch.trace(Sigma_k).item()
        sigmas[d_name] = Sigma_k
        sigmas_norm.append(Sigma_k / tr_k)
        traces[d_name] = tr_k
        print(f"  -> {d_name:<16}: {N_k} estados | Norma Media do Erro ||delta||: {math.sqrt(tr_k):.4f} | Tr(Sigma): {tr_k:.4f}")

    del l63_teacher, l63_atlas
    torch.cuda.empty_cache()

    # Salvar cache dos tensores de erro
    torch.save({k: v.cpu() for k, v in errors.items()}, err_cache_file)
    print(f"  -> Tensores de erro salvos em: {err_cache_file}")

    # 4. Decomposicao Espectral Individual e Media
    print("\n[Passo 4/6] Executando analise espectral (Eigendecomposition)...")
    eigenspectra = {}
    V_dict = {}

    for d_name, Sigma_k in sigmas.items():
        # torch.linalg.eigh retorna autovalores em ordem crescente
        evals, evecs = torch.linalg.eigh(Sigma_k)
        evals_desc = torch.flip(evals, dims=[0])
        evecs_desc = torch.flip(evecs, dims=[1])
        eigenspectra[d_name] = evals_desc.cpu()
        V_dict[d_name] = evecs_desc
        top5_pct = (evals_desc[:5].sum() / traces[d_name]).item() * 100
        top32_pct = (evals_desc[:32].sum() / traces[d_name]).item() * 100
        top64_pct = (evals_desc[:64].sum() / traces[d_name]).item() * 100
        top128_pct = (evals_desc[:128].sum() / traces[d_name]).item() * 100
        print(f"  -> {d_name:<16}: Top-5={top5_pct:.1f}% | Top-32={top32_pct:.1f}% | Top-64={top64_pct:.1f}% | Top-128={top128_pct:.1f}%")

    # Covariancia Media
    Sigma_bar = torch.stack(list(sigmas.values()), dim=0).mean(dim=0)
    tr_bar = torch.trace(Sigma_bar).item()
    evals_bar, evecs_bar = torch.linalg.eigh(Sigma_bar)
    evals_avg_desc = torch.flip(evals_bar, dims=[0])
    U_avg_all = torch.flip(evecs_bar, dims=[1])
    eigenspectra["Average"] = evals_avg_desc.cpu()

    # 5. Otimizacao Minimax e Curvas de Retencao rho(r)
    print("\n[Passo 5/6] Resolvendo Subespacos Minimax no Grassmanniano e Curvas de Retencao...")
    ranks_to_test = [8, 16, 32, 64, 128, 256]

    retention_curves = {
        "ranks": ranks_to_test,
        "domains": {},
        "rho_avg": [],
        "rho_worst_avg": [],
        "rho_worst_minimax": [],
        "minimax_gains": []
    }

    for d_name in domains.keys():
        retention_curves["domains"][d_name] = []

    minimax_bases = {}
    minimax_histories = {}

    header = f"{'Rank':<6} | {'WikiText':<10} | {'HF_Code':<10} | {'GSM8k/D3':<10} | {'rho_avg':<10} | {'Worst(U_avg)':<14} | {'Worst(U_minimax)':<16} | {'Ganho Minimax':<14}"
    print("-" * 100)
    print(header)
    print("-" * 100)

    for r in ranks_to_test:
        # 1. Retencao individual de cada dominio
        for d_name in domains.keys():
            ev = eigenspectra[d_name][:r].sum().item()
            rho_k = (ev / traces[d_name]) * 100.0
            retention_curves["domains"][d_name].append(rho_k)

        # 2. Retencao media rho_avg(r)
        ev_avg = evals_avg_desc[:r].sum().item()
        rho_avg_val = (ev_avg / tr_bar) * 100.0
        retention_curves["rho_avg"].append(rho_avg_val)

        # 3. Pior caso usando a base media U_avg(r)
        U_avg_r = U_avg_all[:, :r]
        worst_avg_val = min([torch.trace(torch.matmul(U_avg_r.t(), torch.matmul(S_norm, U_avg_r))).item() for S_norm in sigmas_norm]) * 100.0
        retention_curves["rho_worst_avg"].append(worst_avg_val)

        # 4. Resolver base minimax U_minimax(r)
        U_minimax_r, hist = solve_minimax_subspace(sigmas_norm, r, U_avg_r, max_iters=50, lr=0.04)
        worst_minimax_val = hist[-1]["worst_retention"] * 100.0
        retention_curves["rho_worst_minimax"].append(worst_minimax_val)
        minimax_gain = worst_minimax_val - worst_avg_val
        retention_curves["minimax_gains"].append(minimax_gain)

        minimax_bases[r] = U_minimax_r.cpu()
        minimax_histories[r] = hist

        d_vals = [retention_curves["domains"][dn][-1] for dn in domains.keys()]
        print(f"r={r:<4} | {d_vals[0]:<9.2f}% | {d_vals[1]:<9.2f}% | {d_vals[2]:<9.2f}% | {rho_avg_val:<9.2f}% | {worst_avg_val:<13.2f}% | {worst_minimax_val:<15.2f}% | {minimax_gain:<+13.2f}%")

    print("-" * 100)

    # 6. Retorno Marginal por Bit R_bits(r)
    print("\n[Passo 6/6] Calculando Retorno Marginal por Bit R_bits(r)...")
    returns_per_bit = []
    bit_costs = []
    # P(r) = 2 * 5120 * r parametros em BF16 (16 bits por param)
    bits_prev = 0
    rho_prev = 0.0

    print(f"{'Rank':<6} | {'Parametros (L63)':<18} | {'Bits (MB)':<12} | {'Delta rho (%)':<14} | {'Retorno por Bit (x10^-6)':<24}")
    print("-" * 80)
    for i, r in enumerate(ranks_to_test):
        params_r = 2 * d_model * r
        bits_r = params_r * 16
        delta_bits = bits_r - bits_prev
        delta_rho = retention_curves["rho_avg"][i] - rho_prev
        r_bit = (delta_rho / delta_bits) * 1e6 if delta_bits > 0 else 0.0

        returns_per_bit.append(r_bit)
        bit_costs.append(bits_r / (8 * 1024 * 1024)) # MB

        print(f"r={r:<4} | {params_r:<18,d} | {bits_r/(8*1024*1024):<10.2f} MB | {delta_rho:<+13.2f}% | {r_bit:<23.4f}")
        bits_prev = bits_r
        rho_prev = retention_curves["rho_avg"][i]

    # Salvar resultados em JSON
    results_json_path = os.path.join(repo_root, "experiments", "uem_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "domains": list(domains.keys()),
            "traces": traces,
            "ranks": ranks_to_test,
            "retention_curves": retention_curves,
            "returns_per_bit": returns_per_bit,
            "bit_costs_mb": bit_costs
        }, f, indent=2)
    print(f"\n[OK] Resultados espectrais exportados para: {results_json_path}")

    # Salvar bases minimax e media
    bases_save_path = os.path.join(repo_root, "checkpoints", "uem_bases_l63.pt")
    torch.save({
        "U_avg_all": U_avg_all.cpu(),
        "U_minimax": minimax_bases
    }, bases_save_path)
    print(f"[OK] Bases U_avg e U_minimax salvas em: {bases_save_path}")

    # 7. Geracao dos Graficos Analiticos (PNG)
    print("\nGerando graficos cientificos publicaveis...")

    # Plot 1: Autovalores (Eigenspectrum)
    plt.figure(figsize=(10, 6), dpi=150)
    for d_name in domains.keys():
        ev = eigenspectra[d_name][:256].numpy()
        plt.semilogy(range(1, 257), ev / traces[d_name], label=f"{d_name} (Normalizado)", lw=1.8)
    ev_a = eigenspectra["Average"][:256].numpy()
    plt.semilogy(range(1, 257), ev_a / tr_bar, label="Covariância Média (Pool)", color="black", lw=2.5, ls="--")
    plt.title("Universal Error Manifold (UEM): Decaimento dos Autovalores do Erro ($L_{63}$)", fontsize=13, pad=12)
    plt.xlabel("Índice do Autovalor (Modo Direcional)", fontsize=11)
    plt.ylabel("Variância Explicada Relativa (log-scale)", fontsize=11)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(repo_root, "uem_eigenspectrum.png")
    plt.savefig(p1)
    plt.close()
    print(f"  -> Salvo: {p1}")

    # Plot 2: rho(r) vs Rank
    plt.figure(figsize=(10, 6), dpi=150)
    for d_name in domains.keys():
        plt.plot(ranks_to_test, retention_curves["domains"][d_name], marker="o", lw=2.0, label=f"$\\rho(r)$ {d_name}")
    plt.plot(ranks_to_test, retention_curves["rho_avg"], marker="s", lw=2.5, color="black", ls="--", label="$\\rho_{\\rm avg}(r)$ (Média)")
    plt.axhline(90, color="gray", ls=":", alpha=0.7, label="90% do Erro Capturado")
    plt.axhline(95, color="gray", ls="-.", alpha=0.7, label="95% do Erro Capturado")
    plt.title("Universal Error Manifold (UEM): Curvas de Retenção de Erro $\\rho(r)$ por Domínio", fontsize=13, pad=12)
    plt.xlabel("Posto do Refinador ($r$)", fontsize=11)
    plt.ylabel("Retenção Acumulada do Erro (%)", fontsize=11)
    plt.xticks(ranks_to_test)
    plt.ylim(0, 102)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(repo_root, "uem_rho_vs_rank.png")
    plt.savefig(p2)
    plt.close()
    print(f"  -> Salvo: {p2}")

    # Plot 3: Pior Caso (Worst-Case) U_avg vs U_minimax
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(ranks_to_test, retention_curves["rho_worst_avg"], marker="o", lw=2.2, color="#d95f02", label="Pior Caso com $U_{\\rm avg}$ (Média)")
    plt.plot(ranks_to_test, retention_curves["rho_worst_minimax"], marker="^", lw=2.5, color="#1b9e77", label="Pior Caso com $U_{\\rm minimax}^*$ (Otimizado)")
    plt.title("Universal Error Manifold (UEM): Comparação de Robustez no Pior Caso", fontsize=13, pad=12)
    plt.xlabel("Posto do Refinador ($r$)", fontsize=11)
    plt.ylabel("Retenção no Pior Domínio $\\rho_{\\rm worst}(r)$ (%)", fontsize=11)
    plt.xticks(ranks_to_test)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    p3 = os.path.join(repo_root, "uem_worstcase_vs_rank.png")
    plt.savefig(p3)
    plt.close()
    print(f"  -> Salvo: {p3}")

    # Plot 4: Retorno Marginal por Bit R_bits(r)
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(ranks_to_test, returns_per_bit, marker="D", lw=2.5, color="#7570b3")
    plt.title("Universal Error Manifold (UEM): Retorno Marginal de Fidelidade por Bit", fontsize=13, pad=12)
    plt.xlabel("Posto do Refinador ($r$)", fontsize=11)
    plt.ylabel("Retorno Marginal $\\mathcal{R}_{\\rm bits}(r) = \\Delta\\rho / \\Delta\\rm Bits$ ($\\times 10^{-6}$)", fontsize=11)
    plt.xticks(ranks_to_test)
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    p4 = os.path.join(repo_root, "uem_return_per_bit.png")
    plt.savefig(p4)
    plt.close()
    print(f"  -> Salvo: {p4}")

    print("\n[OK] Analise UEM 1/3 (Espectral e Minimax) concluida com sucesso!")

if __name__ == "__main__":
    main()
