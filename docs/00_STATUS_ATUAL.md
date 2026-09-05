# Status Técnico Atual do Projeto MathQwen (Setembro de 2026)

## 1. Visão Geral da Mudança de Rota Arquitetural

O projeto MathQwen passou por uma bifurcação metodológica e científica fundamental:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │              Checkpoint Qwen/Qwen3.8-27B-FP8                │
                  │       (64 camadas, d=5120, vocabulário real 248.320)        │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    [PROTÓTIPO LEGADO 9B v1]                                    [ATLAS MATHQWEN CANÔNICO]
  - Isolado em `legacy/g_qwen_9b_v1/`                         - Implementado em `atlas/` e `reference/`
  - Vocabulário simplificado: 152.064                         - Vocabulário real completo: 248.320
  - Embeddings amarrados (tied)                               - Embeddings e lm_head separados
  - DeltaNet ingênua causal (model/layers.py)                 - DeltaNet recorrente oficial (Qwen3_5DecoderLayer)
  - Orçamento teórico restrito a 8.64B                        - Atlas Assimétrico (V_joint, r=2048, ~14.29B)
                                                              - Estabilizadores Residuais (r=64, 41.94M)
                                                              - PPL WikiText-2 reduzida de 3712 -> 39.70
```

---

## 2. O que Está em Cada Diretório

| Diretório | Função | Status |
| :--- | :--- | :---: |
| `legacy/g_qwen_9b_v1/` | Contém o código original do protótipo 9B inicial. Preservado para histórico e reprodutibilidade, mas **não** utilizado na inferência atual. | **Legado** |
| `reference/` | Pipeline determinístico oficial do Qwen 27B-FP8. Carrega os tensores reais via `safetensors`, desquantiza blocos FP8 e executa `Qwen3_5DecoderLayer` com RoPE. | **Canônico Oficial** |
| `atlas/` | Implementação oficial da compressão via Atlas Assimétrico ($V_{\rm joint}$, $r=2048$) e dos módulos de estabilização do residual stream. | **Canônico Oficial** |
| `experiments/` | Scripts empíricos organizados em `layer_probes/`, `stabilization/` e `benchmarks/`. | **Ativo** |
| `tests/` | Testes de equivalência oficial, bases de projeção e testes end-to-end de PPL pelas 64 camadas. | **Ativo** |
| `formal/` | Formalização em Lean 4 do orçamento estrutural de parâmetros e identidades aritméticas de postos. | **Auditado** |

---

## 3. Descobertas Científicas Consolidadas

1. **A Descoberta do Residual Stream**: O Atlas Assimétrico zero-shot ($r=2048$) gera divergência na variedade residual acumulada ($\text{PPL} = 3.712,64$). Uma correção linear compacta no residual stream ($d_{\rm model} = 5120$) de apenas **41,94M parâmetros** ($r=64$, $< 0,3\%$ da rede) reduz a PPL para **39,70** (queda de **98,97%**).
2. **Amplificação Direcional e Ruptura em $L_{48}\text{--}L_{55}$**: As camadas intermediárias acumulam perturbações com ganho secante $G_l^{\rm drift} > 1,0$, culminando em uma duplicação da norma $\|h\|$ na região $L_{48}\text{--}L_{55}$.
3. **Não-Linearidade Residual e Rollback Lexicográfico**: A introdução de não-linearidade suave ($\text{GELU}$) supera o SVD linear em erro de representação ($e_h$) em todas as 8 camadas críticas. O critério de rollback lexicográfico garante que o modelo só aceite corretores não-lineares quando houver redução simultânea de $e_h$ e $\text{NLL}$.
