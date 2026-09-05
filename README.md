# MathQwen: Compressão e Estabilização Funcional do Qwen 27B

> **Repositório de Pesquisa**: Compressão do checkpoint oficial `Qwen/Qwen3.8-27B-FP8` (64 camadas, $d=5120$) via **Atlas Assimétrico de Subespaços Compartilhados** e **Estabilizadores Compactos do Residual Stream**.

---

## 📌 Status Atual da Pesquisa e Arquitetura

O projeto organiza-se em duas vertentes explicitamente separadas:
1. **`atlas/` e `reference/` (Canônico Atual)**:
   - Baseado no checkpoint oficial `Qwen/Qwen3.8-27B-FP8` com vocabulário real completo ($V=248.320$), embeddings e `lm_head` separados.
   - **Atlas Assimétrico ($V_{\rm joint}$, $r=2048$)**: Redução dos pesos densos para ~14,29 bilhões de parâmetros congelados através de 16 cartas locais.
   - **Estabilizadores do Residual Stream ($r=64$, 41,94M parâmetros)**: Redução de **98,97%** na perplexidade no WikiText-2 (de **3.712,64 $\to$ 39,70**).
   - **Sondagem Não-Linear (GELU)**: Incorporação de expressividade dependente do estado com rollback lexicográfico contra a degradação de NLL.
2. **`legacy/g_qwen_9b_v1/` (Protótipo Legado 9B)**:
   - Protótipo inicial com vocabulário simplificado de 152.064 tokens, embeddings amarrados e operador DeltaNet linear preliminar. Preservado para documentação e histórico.

---

## 📂 Estrutura do Repositório

```text
MathQwen/
├── reference/                 # Pipeline oficial de referência (Qwen3_5 oficial, FP8, rotary)
│   ├── loader.py              # Loader determinístico via safetensors e Qwen3_5DecoderLayer
│   └── causal_stream.py       # Loop causal de 64 camadas
│
├── atlas/                     # Implementação oficial do Atlas Assimétrico e Estabilizadores
│   ├── config.py              # Configuração canônica (V=248.320, d=5120, 64 camadas)
│   ├── projection.py          # Álgebra exata das bases V_joint, U_mix, U_down
│   ├── residual.py            # Módulos SVDLinear e NonLinearGELU com solver fechado
│   └── atlas_model.py         # Modelo de inferência e streaming
│
├── experiments/
│   ├── layer_probes/          # Sondagens detalhadas de divergência e camadas críticas
│   ├── stabilization/         # Sweeps de posto residual (r=32..128) e análise de Lipschitz
│   └── benchmarks/            # Resultados consolidados de PPL no WikiText-2
│
├── tests/
│   ├── reference_equivalence/ # Teste de equivalência automática com o Qwen oficial
│   ├── atlas_projection/      # Testes unitários das bases V_joint
│   └── end_to_end/            # Teste end-to-end de 64 camadas e PPL final
│
├── docs/
│   ├── 00_STATUS_ATUAL.md     # Documento mestre de alinhamento metodológico
│   ├── 01 a 11                # Relatórios técnicos anteriores
│   └── 12_ESTABILIZACAO_RESIDUAL_E_SONDAGEM_CRITICA.md # Relatório da estabilização residual
│
├── formal/                    # Formalização em Lean 4 de orçamentos de parâmetros
└── legacy/g_qwen_9b_v1/       # Código original do protótipo 9B v1
```

---

## 🔬 Resultados Principais no WikiText-2 (Teste Cego Não-Visto)

| Modelo | Parâmetros Adicionais | $e_{64}$ | $\cos(z)$ | Top-1 Agree | PPL (WikiText-2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Professor Oficial FP8** | 0 (Referência) | 0.00% | 1.0000 | 100.0% | **4.21** |
| **Atlas Raw ($r=2048$)** | 0 M | 101.53% | 0.7283 | 8.33% | **3712.64** |
| **Escalar ($h' = \alpha_l h$)** | 64 floats | 94.12% | 0.7714 | 16.67% | **649.27** |
| **SVD Low-Rank ($r=32$)** | +20.97 M | 89.41% | 0.8897 | 42.06% | **50.72** |
| **SVD Low-Rank ($r=64$)** | **+41.94 M** | **88.30%** | **0.9075** | **46.83%** | **39.70** |
| **SVD Low-Rank ($r=128$)** | +83.89 M | 87.12% | 0.9175 | 50.00% | **38.22** |

Para detalhes completos dos experimentos e fundamentação matemática, consulte [docs/00_STATUS_ATUAL.md](docs/00_STATUS_ATUAL.md) e [docs/12_ESTABILIZACAO_RESIDUAL_E_SONDAGEM_CRITICA.md](docs/12_ESTABILIZACAO_RESIDUAL_E_SONDAGEM_CRITICA.md).
