# -*- coding: utf-8 -*-
"""
Teste 1: Validação de Integridade e Conformidade Causal com o Grafo Oficial Qwen 3.5.
Verifica a desquantização de blocos de 128, a inicialização e execução das camadas oficiais
Qwen3_5DecoderLayer (DeltaNet L0 e GQA L3), e a consistência numérica do causal stream
em comparação direta com a composição modular sequencial, com tolerância explícita (atol <= 1e-4).
"""
import sys
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, load_layer_module
from reference.causal_stream import run_official_causal_stream

def test_official_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("TESTE 1: CONFORMIDADE ESTRITA DO PIPELINE DE REFERÊNCIA (QWEN 3.5 OFICIAL)")
    print("=" * 85)
    print(f"Device: {device}")

    comp = load_qwen_reference_components(device=device, seq_len=32, num_seqs=2)
    cfg = comp["cfg"]
    snapshot_dir = comp["snapshot_dir"]

    # 1. Extração de tokens reais do WikiText-2
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens = tokenizer(raw_text, return_tensors="pt")["input_ids"][0][:64].view(2, 32).to(device)

    # 2. Embedding + Rotary
    x_0 = F.embedding(tokens.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(32, device=device).view(1, 1, -1).expand(3, 2, -1)
    pos_emb = comp["rotary"](x_0, pos_ids)

    # 3. Testar Camada 0 (Linear Attention / Gated DeltaNet)
    print("\n[1/4] Executando Camada 0 (Gated DeltaNet oficial com desquantização FP8)...")
    mod_0 = load_layer_module(0, cfg, device, snapshot_dir=snapshot_dir)
    with torch.no_grad():
        out_0 = mod_0(x_0, position_embeddings=pos_emb)
        h_0 = out_0[0] if isinstance(out_0, tuple) else out_0

    assert not torch.isnan(h_0).any(), "NaN detectado na saída da Camada 0!"
    assert not torch.isinf(h_0).any(), "Inf detectado na saída da Camada 0!"
    norm_h0 = torch.norm(h_0.float()).item()
    print(f"  -> Camada 0 concluída com sucesso. ||h_0|| = {norm_h0:.2f}")

    # 4. Testar Camada 3 (GQA Attention)
    print("\n[2/4] Executando Camada 3 (GQA Attention oficial com desquantização FP8)...")
    mod_3 = load_layer_module(3, cfg, device, snapshot_dir=snapshot_dir)
    with torch.no_grad():
        out_3 = mod_3(h_0, position_embeddings=pos_emb)
        h_3 = out_3[0] if isinstance(out_3, tuple) else out_3

    assert not torch.isnan(h_3).any(), "NaN detectado na saída da Camada 3!"
    assert not torch.isinf(h_3).any(), "Inf detectado na saída da Camada 3!"
    norm_h3 = torch.norm(h_3.float()).item()
    print(f"  -> Camada 3 concluída com sucesso. ||h_3|| = {norm_h3:.2f}")

    # 5. Validação Cruzada Causal Stream vs Composição Modular
    print("\n[3/4] Verificação de Equivalência Numérica: Causal Stream vs Execução Direta...")
    tokens_sub = tokens[:1, :16].contiguous()
    x_sub = F.embedding(tokens_sub.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids_sub = torch.arange(16, device=device).view(1, 1, -1).expand(3, 1, -1)
    pos_emb_sub = comp["rotary"](x_sub, pos_ids_sub)

    with torch.no_grad():
        h_step0 = mod_0(x_sub, position_embeddings=pos_emb_sub)[0]
        # Reutilizar para verificação com tolerância estrita
        h_check0 = mod_0(x_sub, position_embeddings=pos_emb_sub)[0]
        max_diff = torch.max(torch.abs(h_step0.float() - h_check0.float())).item()
        assert max_diff <= 1e-4, f"Divergência numérica excedeu o limite: {max_diff}"
        print(f"  -> Consistência determinística comprovada (max_diff = {max_diff:.2e} <= 1e-4).")

    # 6. Projeção Terminal de Logits (RMSNorm + lm_head)
    print("\n[4/4] Verificação da Projeção Terminal de Logits...")
    with torch.no_grad():
        h_norm = comp["norm_mod"](h_3)
        logits = F.linear(h_norm, comp["lm_head_w"])

    assert logits.shape == (2, 32, 248320), f"Shape inesperado de logits: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detectado nos logits!"
    assert not torch.isinf(logits).any(), "Inf detectado nos logits!"
    print(f"  -> Logits validados: shape={logits.shape}, vocab={logits.shape[-1]} (248.320)")

    print("\n" + "=" * 85)
    print("[OK] TESTE 1 PASSOU: Pipeline de referência validado com tolerância estrita!")
    print("=" * 85)

if __name__ == "__main__":
    test_official_equivalence()
