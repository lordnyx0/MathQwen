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
│   ├── calibration.py         # Calibração offline e seleção adaptativa com split triplo
│   ├── export.py              # Exportador autônomo compacto em FP8 bloco-128
│   ├── autonomous_model.py    # Runtime autônomo desacoplado do HuggingFace
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


### 🚀 Modelo Atlas Autônomo e Desacoplado (FP8 Bloco-128)

O modelo comprimido final reside em checkpoints/atlas_autonomous/ e executa de forma **100% independente do snapshot de 27B original**:

| Configuração | Formato / Checkpoint | Tamanho em Disco | Pico VRAM | Throughput | NLL Terminal | PPL Terminal | Top-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Professor Oficial FP8** | HF Hub Oficial | 28,31 GB | > 18 GB | - | 1.9892 | 7.31 | 53.87% |
| **Atlas Raw (=2048$)** | HF Hub + Projeção | 28,31 GB | ~7.2 GB | - | 7.6983 | 2204.61 | 7.34% |
| **Atlas Stream (Persistido)** | HF Hub + bases.pt | 29,35 GB | ~5.8 GB | 2.9 tok/s | 5.8730 | 355.31 | 18.75% |
| **Atlas Autônomo (FP8 Bloco-128)** | **Dedicado (tlas_autonomous/)** | **19,57 GB** | **5.02 GB** | **4.7 tok/s** | **6.3065** | **548.10** | **14.78%** |

### Ciclo 24 & 25: A Prova do Upper Bound e a Tese dos 5.120 Eixos

- **Ciclo 24 (Estabilização Profunda $L_0 \to L_{63}$):** Correções lineares estáticas contêm o drift em camadas iniciais ($L_8$: $-15{,}32\%$), mas sofrem **inversão semântica** interdomínios a partir de $L_{32}$, onde circuitos de código e matemática divergem ativamente de atratores estáticos de linguagem geral.
- **Ciclo 25 (Trilho Orbital e Prova do Upper Bound Absoluto):** Extração da órbita contínua em $U_c(32)$ via Procrustes em $\text{SO}(32)$. Teste do Oráculo perfeito no subespaço $U_c(32)$ comprovou que mesmo com erro residual de apenas $0{,}286\%$ em $U_c$, o modelo colapsa para **PPL $2.382{,}18$** (Top-1 $4{,}37\%$).
- **A Tese dos 5.120 Eixos:** $\dim(U_c) = 32$ representa apenas $0{,}625\%$ da variedade. Os $5.088$ eixos perpendiculares descontrolados dominam os produtos bilineares de Atenção e as expansões multiplicativas do SwiGLU ($5.120 \to 18.944$), tornando matematicamente impossível recuperar o raciocínio semântico profundo sem preservar a totalidade dos $5.120$ eixos.
- **Transição para o Caminho 4:** Pesquisas de correções residuais de baixo rank definitivamente concluídas e encerradas. O foco avança para a preservação de todos os 5.120 eixos via quantização 4-bit (NF4/TorchAO) com KV-cache persistente para geração em alta velocidade (<200 ms/token) na RTX 3060 12GB.
