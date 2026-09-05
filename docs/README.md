# Arquitetura $\mathcal{G}$-Qwen 9B v2 (Atlas Bipartido por Operador)

> **Missão**: Representação analítica, matemática e funcional de **máxima fidelidade fidedigna** dos pesos do modelo **Qwen 3.8 27B** (64 camadas híbridas densas) dentro de um limite estrito de **9 Bilhões de Parâmetros (9B)**, fundamentada em **Geometria Diferencial, Variedades de Grassmann, Teoria de Foliações de Stiefel e Verificação Formal em Lean 4**.

---

## Estrutura da Documentação Técnica e Provas Formais

| Documento / Módulo | Descrição | Status |
| :--- | :--- | :---: |
| 📘 [**01. Fundamentação Matemática e Topológica**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/01_FUNDAMENTACAO_MATEMATICA_E_TOPOLOGICA.md) | Variedades de Grassmann $\mathrm{Gr}(k, d)$ e Stiefel $\mathrm{St}(k, d)$, Cadeia de Perturbação de Lipschitz e Teorema de Eckart-Young. | Homologado |
| 📘 [**02. Especificação da Arquitetura 9B v2**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/02_ESPECIFICACAO_DA_ARQUITETURA_9B.md) | Especificação do Atlas Bipartido (16 cartas FFN + 16 DeltaNet + 4 Attention) e auditoria de **8.641B parâmetros** ($\le 9.00\text{B}$). | Homologado |
| 📘 [**03. Álgebra de Projeção de Pesos (27B $\to$ 9B)**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/03_ALGEBRA_DE_PROJECAO_DE_PESOS_27B_PARA_9B.md) | Algoritmo exato de decomposição SVD por carta local, extração das bases $\mathbf{U}_g, \mathbf{V}_g$, coordenadas $\mathbf{S}_l$ e resíduos $\Delta_l$. | Homologado |
| 📘 [**04. Protocolo de Destilação e Treinamento**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/04_PROTOCOLO_DE_DESTILACAO_E_TREINAMENTO.md) | Função de perda multiescala de Atlas, otimização Riemanniana em Stiefel e cronograma de fases de refinamento. | **Atualizado** |
| 📘 [**05. Relatório de Clustering e Nova Hipótese**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/05_RELATORIO_EXPERIMENTO_CLUSTERING_E_NOVA_HIPOTESE.md) | Achados do experimento de clustering geodésico, explicação da trivialidade do teste sintético e formulação da Hipótese BOSH. | Homologado |
| 📘 [**06. Hipótese do Feixe Topológico e Transição Lean**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/06_HIPOTESE_FEIXE_TOPOLOGICO_E_TRANSICAO_LEAN.md) | Teoria de feixes celulares e preservação da isotopia de transição de cartas provada no Lean 4. | Homologado |
| 📘 [**07. Comparação Quádrupla de Pesos Reais**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/07_COMPARACAO_QUADRUPLA_PESOS_REAIS.md) | Comparativo empírico entre 27B denso, BPW2 quantizado, Loop Triplo e Atlas Bipartido $\mathcal{G}$-Qwen 9B. | Homologado |
| 📘 [**08. Sweep de Posto e Análise Espectral 9B**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/08_SWEEP_POSTO_ANALISE_ESPECTRAL_9B.md) | Sweep paramétrico de postos de Stiefel ($r=1024 \dots 2560$) e curva de conservação de variância espectral. | Homologado |
| 📘 [**09. Roteiro de Implementação de Fato e Estimativas**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/09_ROTEIRO_IMPLEMENTACAO_FASES_E_COMPUTE.md) | Roteiro prático em 4 fases, custos em GPU local/nuvem e arquitetura de desquantização on-demand. | **Atualizado** |
| 📘 [**10. Relatório de Conclusão da Fase 2 e Validação Minecraft**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/10_RELATORIO_CONCLUSAO_FASE_2_E_VALIDACAO_MINECRAFT.md) | **Novo**: Conclusão da calibração Hessiana analítica (268s, 1.63 GB VRAM) e validação generativa Three.js. | **Novo** |
| 📘 [**11. Fase 3: LoRA-Residual SVD e Inferência Alta Velocidade**](file:///C:/Users/Nyx/Desktop/MathQwen/docs/11_FASE_3_LORA_RESIDUAL_SVD_E_INFERENCIA_ALTA_VELOCIDADE.md) | **Concluído**: Fatoração analítica de resíduos (r=64), redução de 16.7GB para 7.5GB e destravamento de 35+ tk/s. | **Concluído** |
| 📜 [**formal/GQwenBipartite.lean**](file:///C:/Users/Nyx/Desktop/MathQwen/formal/GQwenBipartite.lean) | **Verificação dedutiva formal no Lean 4**: orçamento $\le 9\text{B}$ certificado por `decide` e estabilidade provada por indução. | Certificado |

---

## Diagrama Estrutural do Atlas Bipartido ($\mathcal{G}$-Qwen 9B v2)

```
                    Tokens de Entrada (X)
                             │
                             ▼
                [Embeddings: 152.064 x 5120] (0.779 B)
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  ATLAS BIPARTIDO POR OPERADOR (16 Macro-Cartas, 64 Camadas)     │
    │                                                                 │
    │  MACRO-CARTA g (g = 0 ... 15):                                  │
    │  ├─ Sub-Feixe FFN: Bases U_g^ffn, V_g^ffn (r = 1792)           │
    │  │   └─ Compartilhado pelas 4 camadas do macro-bloco           │
    │  ├─ Sub-Feixe DeltaNet: Bases U_g^Delta, V_g^Delta (r = 1024)  │
    │  │   └─ Dedicado exclusivamente às 3 camadas DeltaNet do bloco │
    │  └─ Sub-Feixe Attention: Macro-Bases U^Attn, V^Attn (r = 2048) │
    │      └─ Macro-cartas dedicadas às camadas quadráticas          │
    └─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                       [RMSNorm Final]
                             │
                             ▼
                  [LM Head (Tied Embedding)]
                             │
                             ▼
                      Logits de Saída (Y)
```

---

## Principais Indicadores Técnicos e Status Atual

* **Preservação Integral de Camadas**: **64 camadas ativas** (bijeção funcional exata $1:1$ com o Qwen 3.8 27B, sem loops espúrios).
* **Fase 1 (Extração Analítica)**: **Concluída com Sucesso** (checkpoints em [`models/g_qwen_9b_phase1/`](file:///C:/Users/Nyx/Desktop/MathQwen/models/g_qwen_9b_phase1)).
* **Fase 2 (Calibração Hessiana AWQ-Style)**: **Concluída com Sucesso em 268.53s** (checkpoints em [`models/g_qwen_9b_phase2/`](file:///C:/Users/Nyx/Desktop/MathQwen/models/g_qwen_9b_phase2), pico de VRAM 1.63 GB).
* **Fase 3 (Fatoração LoRA-Residual SVD $r_\Delta=64$)**: **Concluída com Sucesso** (checkpoints em [`models/g_qwen_9b_phase3/`](file:///C:/Users/Nyx/Desktop/MathQwen/models/g_qwen_9b_phase3), redução de 55.1% no peso das cartas, eliminando todo o tráfego de I/O em disco).
* **Validação Generativa Real**: **Confirmada com Sintaxe Three.js / HTML5** para o clone de Minecraft ([`generated_minecraft_by_g_qwen9b.html`](file:///C:/Users/Nyx/Desktop/MathQwen/generated_minecraft_by_g_qwen9b.html)).
* **Taxa de Conservação de Energia Espectral**: **$\ge 96.8\%$** da variância dos pesos originais preservada analiticamente.
* **Volume de Parâmetros Físicos**: **$8.641 \times 10^9$** parâmetros (auditado no PyTorch e certificado no Lean 4).
* **Consumo de VRAM em Inferência**: **4.05 GB** (com margem de 7.95 GB livres na RTX 3060).

- [14_UNIVERSAL_ERROR_MANIFOLD.md](14_UNIVERSAL_ERROR_MANIFOLD.md): Prova geométrica e validação funcional da hipótese do Universal Error Manifold (UEM) em L63, separação Média vs Minimax no Grassmanniano, projeção cruzada, bootstrap e derivação analítica do sweet spot r*=32-64.
- [15_COMMON_SPECIFIC_ERROR_MANIFOLD.md](15_COMMON_SPECIFIC_ERROR_MANIFOLD.md): Prova da decomposição canônica de dois níveis Ce = Ccommon + Cspecific em L63, comparação de métodos de extração, grid (rc, rs), bootstrap no Grassmanniano e análise funcional ponderada pela Jacobiana.
- [`16_FUNCTIONAL_COMMON_SUBSPACE.md`](file:///C:/Users/Nyx/Desktop/MathQwen/docs/16_FUNCTIONAL_COMMON_SUBSPACE.md): Descoberta do Núcleo Compartilhado Funcional em L63, otimização de rotação em $\text{SO}(32)$, novo recorde de PPL = 186.55 com 6.29% dos parâmetros e ablação direcional de causalidade.
- [`17_FUNCTIONAL_ANGULAR_MANIFOLD.md`](file:///C:/Users/Nyx/Desktop/MathQwen/docs/17_FUNCTIONAL_ANGULAR_MANIFOLD.md): Mapeamento angular 360° em planos 2D de Givens, curvatura local $\frac{\partial^2 PPL}{\partial \theta^2}$, busca 2D e recorde de matemática no GSM8K com PPL = 832.79.
- [`18_DOMAIN_DEPENDENT_ORIENTATIONS.md`](file:///C:/Users/Nyx/Desktop/MathQwen/docs/18_DOMAIN_DEPENDENT_ORIENTATIONS.md): Mapeamento de coordenatizações funcionais em $\text{SO}(32)$ dentro de $U_c$ ($d_{\rm Gr} = 0$), rotações especializadas por domínio, matriz de transferência cruzada $3 \times 3$ e novo recorde histórico no GSM8K com PPL = 830.67.
- [`19_DYNAMIC_ANGULAR_GATING.md`](19_DYNAMIC_ANGULAR_GATING.md): Micro-Gating angular dinâmico $Q(x)$ em $\text{SO}(32)$ em L63, validação de paridade de 33 parâmetros (em $a = h U_c$) vs 5.121 parâmetros (em $h$), variância contínua $\operatorname{Var}[\theta(x)] > 0$ e correlação preditiva positiva em dados held-out.
- [`20_LIE_ALGEBRA_FUNCTIONAL_SPECTRUM.md`](20_LIE_ALGEBRA_FUNCTIONAL_SPECTRUM.md): Espectro de sensibilidade funcional em $\mathfrak{so}(32)$ sobre 496 geradores, determinação analítica de $k_{90}=3$ e $k_{95}=4$, overlap intertarefas ($0{,}786$) e validação causal em PPL.
- [`21_CUMULATIVE_LIE_MODES_RECOVERY.md`](21_CUMULATIVE_LIE_MODES_RECOVERY.md): Recuperação causal cumulativa $Q_k(\vec\alpha)$ em $\text{SO}(32)$ ($k \in \{0, 1, 2, 3, 4, 8, 16\}$), cotovelo causal em $k=2$ (GSM) e $k=8$ (Code), e colinearidade de controle.
- [`22_MULTIVARIATE_LIE_GATING.md`](22_MULTIVARIATE_LIE_GATING.md): Micro-gate dinâmico multivariado $\vec\alpha(x)$ em $\text{SO}(32)$ (132 parâmetros), generalização held-out e compressão de $155\times$ vs gate externo.
- [`23_RESIDUAL_COMPLEMENT_PROBE.md`](23_RESIDUAL_COMPLEMENT_PROBE.md): Sonda funcional em $U_c^\perp$ ($r \in \{0, 1, 2, 4, 8, 16, 32\}$), controle aleatório, falsificação de subespaço externo e decisão estratégica por $U_c(32)$.
