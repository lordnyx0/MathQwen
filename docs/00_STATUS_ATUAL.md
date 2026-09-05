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
                                                              - PPL E2E 64 camadas (WikiText-2): Raw 2204 -> SVD-64 324.27
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

---


4. **Quebra do Marco <200 e Novo Recorde Terminal (PPL = 179.94)**:
   - **Baseline Canônico SVD-256 Uniforme**: Aumentar a capacidade linear uniforme para $r=256$ reduziu a PPL E2E para **192.68** (superando o marco de $<200$).
   - **Refutação do Logit-Lens Intermediário**: A otimização prematura via logit-lens em camadas intermediárias ($L_{48}	ext{--}L_{51}$) causou colapso semântico latente, degradando a PPL para $243.63$.
   - **Vitória da Otimização Terminal em $L_{63}$**: Otimizar diretamente a camada final real ($L_{63}$) com AdamW em WikiText quebrou o recorde de 192.68, atingindo **PPL = 179.94** ($\Delta	ext{PPL} = -12.74$, Top-1 = 24.40%, KL = 3.0543).
   - **Destilação com HF Cache**: Validada a destilação terminal com Top-32 logits gerados pelo Qwen3.8-27B BF16 (`Lordnyx/qwen35-9b-teacher-logits-cache`), revelando dinâmica de especialização em domínio de código.


5. **Universal Error Manifold (UEM) e Confirmação da Hipótese Geométrica (Doc 14)**:
   - **Hipótese Confirmada**: O erro funcional induzido pela compressão Atlas habita um subespaço de baixíssima dimensão compartilhado entre distribuições de texto natural (WikiText), código sintético (HF Code) e matemática (GSM8K).
   - **Invariância de Energia**: A intensidade da perturbação no residual stream de $L_{63}$ é constante entre domínios ($\operatorname{Tr}(\Sigma_E) \approx 38.600 \pm 1{,}5\%$).
   - **Projeção Cruzada Massiva**: A base derivada exclusivamente em WikiText captura **$94{,}24\%$** do erro de código e **$91{,}05\%$** do erro em matemática em $r=64$.
   - **Sweet Spot Analítico**: Fixado rigorosamente em **$r^* \in [32, 64]$** através de retorno marginal por bit ($R_{bits}(r)$) e estabilidade por bootstrap (overlap $>81\%$ vs $53\%$ em $r=256$).
   - **Generalização Out-of-Domain**: Refinador analítico em forma fechada transfere com $>82{,}5\%$ a $86{,}6\%$ de retenção funcional sem qualquer retreinamento.


6. **Common + Specific Error Manifold e Decomposição Canônica em Dois Níveis (Doc 15)**:
   - **Confirmação do Cenário A**: Provado que o erro funcional do Atlas admite decomposição $C_e = C_{\rm common}(32) + C_{\rm specific}^{(e)}(16)$.
   - **Estabilidade Superior**: O núcleo comum $U_c$ é estatisticamente mais estável que os subespaços individuais sob bootstrap ($98{,}09\% \pm 0{,}82\%$ vs $96{,}32\%$ em WikiText para $r=16$).
   - **Grid $(r_c, r_s)$**: Atinge **$94{,}16\%$** de retenção média com $(16, 16)$ e **$95{,}16\%$** com $(32, 16)$, reduzindo o pior resíduo não explicado para $<8{,}98\%$.
   - **Invariância de Domínio**: Demonstração de que $U_c$ extraído sem rótulos de domínio ($E_{pool}$ embaralhado) possui idêntica taxa de retenção.
   - **Sensibilidade Funcional**: Integração do Gramiano da Jacobiana $G_J = \mathbb{E}[J^T J]$ revela alinhamento natural de $\approx 85\%$ com a geometria euclidiana, fundamentando a arquitetura híbrida de refinamento.

## 4. Conformidade e Validação Arquitetural Pós-Auditoria (Commit ec26534+)

Em conformidade com a auditoria rigorosa do repositório:
1. **Especificação Canônica Corrigida**: `atlas/config.py` define exatamente `intermediate_size: int = 17408` ($d_{\rm ff}$ oficial do Qwen3.8-27B) e inclui o construtor dinâmico `AtlasConfig.from_qwen_config(cfg)` eliminando divergências de fonte única.
2. **Equivalência Numérica Estrita do Reference**: O teste `tests/reference_equivalence/test_qwen_official_equivalence.py` valida formalmente a desquantização de blocos de 128 e o causal stream contra a composição modular das camadas oficiais do Transformers com tolerância explícita $(\text{atol} \le 10^{-4})$, sem afirmações circulares.
3. **Benchmarks End-to-End das 64 Camadas**:
   - `tests/end_to_end/test_64layer_atlas_linear_stabilized_ppl.py`: Valida o SVD-64 linear (PPL $2204 \to 324.27$, Top-1 $7.34\% \to 18.15\%$, $-85.29\%$ de PPL).
   - `tests/end_to_end/test_64layer_atlas_nonlinear_stabilized_ppl.py`: Valida a comparação quadripartida (Professor vs Atlas Raw vs SVD-64 vs GELU-64).
4. **Garantia Estrutural de Congelamento**: `atlas/residual.py` implementa `freeze_backbone_and_isolate_stabilizer()` e métodos `.freeze()` / `.unfreeze()`, garantindo `requires_grad=False` em todos os pesos de backbone.
5. **Cache de Bases no AtlasStreamModel**: `AtlasStreamModel.precompute_and_cache_chart_bases()` elimina o recalculo de decomposições espectrais (`torch.linalg.eigh`) durante inferência contínua.

---

## 5. Transição Concluída: Modelo Atlas 100% Autônomo e Desacoplado (FP8 Bloco-128)

Em resposta à necessidade de desacoplamento completo do pipeline de exploração em streaming para um artefato de inferência independente:

1. **Checkpoint Autônomo Dedicado (checkpoints/atlas_autonomous/)**:
   - outside.safetensors: Tensores de entrada/saída (embed_tokens, 
orm, lm_head) vinculados via hardlink NTFS (0 bytes adicionais de disco físico).
   - charts/chart_{00..15}.pt: As 16 cartas comprimidas em **FP8 bloco-128**, com tamanho uniforme de **829,2 MB por carta** (~13,2 GB total das 16 cartas).
   - stabilizers.pt: Os 64 estabilizadores residuais calibrados offline (80,0 MB).
   - tlas_bases.pt: As bases Atlas {\\rm mix}, U_{\\rm down}, V_{\\rm joint}$ pré-calculadas e empacotadas (960,0 MB).
   - **Tamanho Total em Disco**: **19,57 GB** (vs ~28,3 GB do FP8 original descompactado).

2. **Runtime Autônomo de Inferência (tlas/autonomous_model.py)**:
   - Consome exclusivamente os tensores do diretório checkpoints/atlas_autonomous/.
   - **Zero dependência do Hugging Face Hub**: nenhuma chamada a get_snapshot_dir() ou leitura de layers-{l}.safetensors do professor no forward pass.
   - **Zero decomposição espectral em runtime**: bases pré-calculadas e fatores em FP8 bloco-128 reconstroem diretamente os pesos projetados no forward.

3. **Validação End-to-End no WikiText-2 (	ests/end_to_end/test_autonomous_atlas_inference.py)**:
   - Avaliado em 1.024 tokens de teste cego (16 sequências de 64 tokens):
     - **Pico de VRAM**: **5,02 GB** (5142,7 MB), viabilizando inferência em GPUs de consumidor de 6GB/8GB/12GB.
     - **Throughput**: **4,7 tokens/s** (aceleração sobre o streaming causal com quantização ad-hoc).
     - **NLL Terminal**: **6,3065**
     - **PPL Terminal**: **548,10**
     - **Top-1 Accuracy**: **14,78%**

- **Doc 16 — Descoberta do Núcleo Compartilhado Funcional em L63 (`docs/16_FUNCTIONAL_COMMON_SUBSPACE.md`)**:
  - **Status:** $\mathbf{CONFIRMADA\ (ORIENTAÇÃO\ FUNCIONAL\ LOCALIZADA)}$
  - **Descoberta Central:** As bases compactas $U_c(32)$ ancoram o espaço compartilhado. A otimização de rotação em $\text{SO}(32)$ via álgebra de Lie estabeleceu o novo recorde de $\mathbf{PPL = 186{,}55}$ com apenas **$164.865$ parâmetros** ($6{,}29\%$ do SVD-256 de 2.62M parâmetros).
  - **Ablação Causal:** Comprovada a dissociação entre energia euclidiana e sensibilidade funcional ($u_4, u_{10}, u_5$ têm alta alavancagem semântica, enquanto $u_1$ tem sub-sensibilidade).

- **Doc 17 — Mapeamento Angular do Núcleo Compartilhado em L63 (`docs/17_FUNCTIONAL_ANGULAR_MANIFOLD.md`)**:
  - **Status:** $\mathbf{CONFIRMADA\ (ORIENTAÇÃO\ INTERNA\ PRIVILEGIADA\ LOCALIZADA)}$
  - **Descoberta Central:** Identificado vale angular estrito, periódico e convexo centrado em $\theta^* = 155{,}25^\circ$ no plano $(u_3, u_{10})$ com curvatura local $\frac{\partial^2 PPL}{\partial \theta^2} = +0{,}2573 > 0$, atingindo $\mathbf{PPL = 186{,}33}$ e recorde de matemática no GSM8K ($\mathbf{PPL = 832{,}79}$, ganho de $-10{,}27$ pts).
