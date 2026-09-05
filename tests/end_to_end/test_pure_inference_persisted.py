# -*- coding: utf-8 -*-
"""
Teste 4: Inferência Pura e Desacoplada do Atlas com Estabilizadores Persistidos.
Demonstra que o AtlasStreamModel executa a inferência de 64 camadas sem:
  - Carregar o modelo professor;
  - Executar calibração/fit em tempo de execução;
  - Decomposição espectral repetida.
Avalia a PPL terminal da política híbrida adaptativa no conjunto cego do WikiText-2.
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

from atlas.config import AtlasConfig
from atlas.atlas_model import AtlasStreamModel
from reference.loader import get_snapshot_dir

def compute_ppl_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    vocab_size = logits.shape[-1]
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
    targets_flat = targets.reshape(-1)
    nll = F.cross_entropy(logits_pred, targets_flat, reduction="mean").item()
    ppl = math.exp(nll)
    top1 = (logits_pred.argmax(dim=-1) == targets_flat).float().mean().item() * 100.0
    return nll, ppl, top1

def test_pure_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print("TESTE 4: INFERÊNCIA PURA DESACOPLADA (ESTABILIZADORES PERSISTIDOS EM DISCO)")
    print("=" * 90)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    checkpoint_path = os.path.join(repo_root, "checkpoints", "atlas_stabilizers_adaptive.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint de estabilizadores não encontrado em: {checkpoint_path}. Execute a calibração offline primeiro.")

    # 1. Carregar tokens reais de teste (1.024 tokens cegos)
    snapshot_dir = get_snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 16
    seq_len = 64
    # Pega exatamente o conjunto cego (tokens de índice 1024 a 2048)
    test_ids = tokens_all[1024:1024 + num_seqs * seq_len].view(num_seqs, seq_len).to(device)
    test_targets = test_ids[:, 1:].contiguous()

    # 2. Inicializar o modelo e carregar os estabilizadores persistidos
    print(f"\n[1/3] Inicializando AtlasStreamModel e carregando estabilizadores de {checkpoint_path}...")
    model = AtlasStreamModel(device=device)
    model.freeze_backbone()
    model.load_stabilizers(checkpoint_path)

    # 3. Execução de inferência pura (100% desacoplada do professor)
    print(f"\n[2/3] Executando streaming puro pelas 64 camadas em {num_seqs} sequências de {seq_len} tokens...")
    t_start = time.time()
    logits = model.forward_tokens(test_ids, use_atlas=True)
    t_total = time.time() - t_start

    total_tokens = num_seqs * seq_len
    throughput = total_tokens / t_total
    print(f"  -> Inferência concluída em {t_total:.2f}s ({throughput:.1f} tokens/s).")

    # 4. Avaliação de NLL, PPL e Top-1 Accuracy
    print(f"\n[3/3] Avaliação de Desempenho no Teste Cego...")
    nll, ppl, top1 = compute_ppl_from_logits(logits, test_targets)

    print("\n" + "=" * 90)
    print("           RESULTADO DA INFERÊNCIA PURA DESACOPLADA (POLÍTICA ADAPTATIVA)")
    print("=" * 90)
    print(f"NLL Terminal:       {nll:.4f}")
    print(f"PPL Terminal:       {ppl:.2f}")
    print(f"Top-1 Accuracy:     {top1:.2f}%")
    print(f"Throughput:         {throughput:.1f} tokens/s")
    print("=" * 90)

    assert not torch.isnan(logits).any(), "NaN detectado nos logits de inferência!"
    assert ppl < 1000.0, f"PPL muito alta ({ppl:.2f}), esperava estabilização efetiva."
    print("\n[OK] TESTE 4 PASSOU: Inferência pura desacoplada validada com sucesso!")

if __name__ == "__main__":
    test_pure_inference()
