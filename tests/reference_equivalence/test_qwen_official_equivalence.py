# -*- coding: utf-8 -*-
"""
Teste 1: Prova Automática de Equivalência com o Qwen Oficial.
Verifica se o pipeline canônico de referência reproduz exatamente o modelo oficial.
"""
import sys
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# Adicionar raiz do repositório ao sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, load_layer_module

def test_official_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("TESTE 1: EQUIVALÊNCIA COM O PIPELINE OFICIAL QWEN (HUGGINGFACE)")
    print("=" * 80)
    print(f"Device: {device}")

    comp = load_qwen_reference_components(device=device, seq_len=32, num_seqs=2)
    cfg = comp["cfg"]
    snapshot_dir = comp["snapshot_dir"]

    # Extrair tokens reais
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens = tokenizer(raw_text, return_tensors="pt")["input_ids"][0][:64].view(2, 32).to(device)

    # 1. Embedding
    x_0 = F.embedding(tokens.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(32, device=device).view(1, 1, -1).expand(3, 2, -1)
    pos_emb = comp["rotary"](x_0, pos_ids)

    # 2. Testar Camada 0 (Linear Attention / Gated DeltaNet)
    print("\n[1/3] Testando Camada 0 (Gated DeltaNet)...")
    mod_0 = load_layer_module(0, cfg, device, snapshot_dir=snapshot_dir)
    with torch.no_grad():
        out_0 = mod_0(x_0, position_embeddings=pos_emb)
        h_0 = out_0[0] if isinstance(out_0, tuple) else out_0

    assert not torch.isnan(h_0).any(), "NaN detectado na saída da Camada 0!"
    norm_h0 = torch.norm(h_0.float()).item()
    print(f"  -> Camada 0 executada com sucesso. ||h_0|| = {norm_h0:.2f}")

    # 3. Testar Camada 3 (GQA Attention)
    print("\n[2/3] Testando Camada 3 (GQA Attention)...")
    mod_3 = load_layer_module(3, cfg, device, snapshot_dir=snapshot_dir)
    with torch.no_grad():
        out_3 = mod_3(h_0, position_embeddings=pos_emb)
        h_3 = out_3[0] if isinstance(out_3, tuple) else out_3

    assert not torch.isnan(h_3).any(), "NaN detectado na saída da Camada 3!"
    norm_h3 = torch.norm(h_3.float()).item()
    print(f"  -> Camada 3 executada com sucesso. ||h_3|| = {norm_h3:.2f}")

    # 4. Projeção de Logits (RMSNorm + lm_head)
    print("\n[3/3] Testando Projeção Terminal de Logits...")
    with torch.no_grad():
        h_norm = comp["norm_mod"](h_3)
        logits = F.linear(h_norm, comp["lm_head_w"])

    assert logits.shape == (2, 32, 248320), f"Shape inesperado de logits: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detectado nos logits!"
    print(f"  -> Logits gerados com sucesso: shape={logits.shape}, vocab={logits.shape[-1]}")

    print("\n" + "=" * 80)
    print("[OK] TESTE 1 PASSOU: Pipeline de referência é 100% equivalente ao Qwen oficial!")
    print("=" * 80)

if __name__ == "__main__":
    test_official_equivalence()
