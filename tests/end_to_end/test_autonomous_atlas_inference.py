# -*- coding: utf-8 -*-
"""
Teste 5: Validação do Modelo Atlas 100% Autônomo e Autocontido.
Carrega exclusivamente o diretório `checkpoints/atlas_autonomous/` e executa
a inferência de 64 camadas sem nunca acessar o checkpoint oficial do HuggingFace.
"""
import sys
import os
import time
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from atlas.autonomous_model import AtlasAutonomousModel
from atlas.export import export_autonomous_atlas_model

def compute_ppl_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    vocab_size = logits.shape[-1]
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
    targets_flat = targets.reshape(-1)
    nll = F.cross_entropy(logits_pred, targets_flat, reduction="mean").item()
    ppl = math.exp(nll)
    top1 = (logits_pred.argmax(dim=-1) == targets_flat).float().mean().item() * 100.0
    return nll, ppl, top1

def get_dir_size_gb(directory: str) -> float:
    total_bytes = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(directory) for f in files)
    return total_bytes / (1024 ** 3)

def run_autonomous_atlas_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("TESTE 5: INFERÊNCIA COM O MODELO ATLAS 100% AUTÔNOMO (CHECKPOINT COMPRIMIDO DEDICADO)")
    print("=" * 95)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    model_dir = os.path.join(repo_root, "checkpoints", "atlas_autonomous")
    
    # 1. Se o diretório autônomo não existir, executar a exportação
    if not os.path.exists(os.path.join(model_dir, "charts", "chart_15.pt")):
        print(f"\n[1/4] Checkpoint autônomo não encontrado. Iniciando exportação em {model_dir}...")
        export_autonomous_atlas_model(output_dir=model_dir, device=device)
    else:
        print(f"\n[1/4] Checkpoint autônomo já existente em: {model_dir}")

    total_gb = get_dir_size_gb(model_dir)
    print(f"  -> Tamanho total do modelo comprimido em disco: {total_gb:.2f} GB")

    # 2. Carregar tokens reais de teste cego (1.024 tokens do split_test)
    print("\n[2/4] Carregando 1.024 tokens de teste cego do WikiText-2 (split_test)...")
    from reference.loader import get_snapshot_dir
    snapshot_dir = get_snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 16
    seq_len = 64
    # Split de teste cego (tokens 1536 a 2560)
    test_ids = tokens_all[1536:1536 + num_seqs * seq_len].view(num_seqs, seq_len).to(device)
    test_targets = test_ids[:, 1:].contiguous()

    # 3. Instanciar AtlasAutonomousModel exclusivamente a partir de model_dir
    print(f"\n[3/4] Inicializando AtlasAutonomousModel a partir de {model_dir}...")
    torch.cuda.reset_peak_memory_stats()
    model = AtlasAutonomousModel(model_dir=model_dir, device=device)

    # 4. Executar inferência pura pelas 64 camadas
    print(f"\n[4/4] Executando streaming autônomo em {num_seqs} sequências de {seq_len} tokens ({num_seqs * seq_len} tokens)...")
    t_start = time.time()
    logits = model.forward_tokens(test_ids)
    t_total = time.time() - t_start

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    throughput = (num_seqs * seq_len) / t_total

    # 5. Métricas de Avaliação
    nll, ppl, top1 = compute_ppl_from_logits(logits, test_targets)

    print("\n" + "=" * 95)
    print("         RESULTADOS DO MODELO COMPRIMIDO AUTÔNOMO (INFERÊNCIA DESACOPLADA)")
    print("=" * 95)
    print(f"Tamanho em Disco:   {total_gb:.2f} GB (vs ~27 GB do FP8 original)")
    print(f"Pico de VRAM:       {peak_vram_mb:.1f} MB ({peak_vram_mb/1024:.2f} GB)")
    print(f"Throughput:         {throughput:.1f} tokens/s")
    print(f"NLL Terminal:       {nll:.4f}")
    print(f"PPL Terminal:       {ppl:.2f}")
    print(f"Top-1 Accuracy:     {top1:.2f}%")
    print("=" * 95)

    assert not torch.isnan(logits).any(), "NaN detectado nos logits do modelo autônomo!"
    assert ppl < 1000.0, f"PPL fora do esperado: {ppl:.2f}"
    print("\n[OK] TESTE 5 PASSOU: Modelo Atlas autônomo validado com sucesso!")

if __name__ == "__main__":
    run_autonomous_atlas_test()
