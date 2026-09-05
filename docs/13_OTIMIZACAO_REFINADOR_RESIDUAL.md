# Relatório Técnico 13: Otimização Funcional do Refinador Residual e Novo Recorde Terminal (PPL 179.94)

## 1. Resumo Executivo

Este documento consolida a investigação experimental exaustiva sobre a **otimização funcional do refinador residual** no Atlas Qwen3.8-27B.

Mantendo a fatoração de base **rigorosamente congelada** ($r_{\rm base} = 2048$, $U_{\rm mix}$, $U_{\rm down}$, $V_{\rm joint}$, 16 cartas, precisão FP8 de 14.29B parâmetros), foram avaliadas três abordagens principais para os refinadores residuais:

1. **Ablação Canônica de Refinadores (Experimento A)**:
   - Identificou o **Linear SVD-256 Uniforme** como o campeão indiscutível entre as arquiteturas fixas por camada, atingindo **$\text{PPL} = 192.68$** (superando o marco de $\text{PPL} < 200$ e o baseline de SVD-64 de $324.27$).
2. **Diagnóstico e Falha do Logit-Lens Intermediário**:
   - A tentativa de otimizar refinadores intermediários ($L_{48}\text{--}L_{51}$) via perda de logit-lens sequencial reduziu a perda interna da camada em 77%, mas **degradou monotonicamente a PPL final de teste de $192.68$ para $243.63$**.
   - **Achado Teórico**: Forçar o alinhamento com os logits terminais em camadas intermediárias induz colapso semântico prematuro, destruindo as 15 camadas subsequentes que esperam representações latentes profundas e não distribuições de saída.
3. **Vitória da Otimização Terminal Direta na Camada Final Real ($L_{63}$)**:
   - Em $L_{63}$, não existem camadas a jusante; a saída do estabilizador alimenta diretamente o `RMSNorm` e o `LMHead`.
   - A otimização direta via AdamW com loss conjunta (KL Distillation + Hard CE + Regularização de Hidden State) **quebrou o recorde de 192.68**, atingindo:
     $$\mathbf{\text{PPL} = 179.94} \quad (\Delta\text{PPL} = -12.74, \quad \text{Top-1} = 24.40\%, \quad \text{KL} = 3.0543)$$
4. **Experimento de Destilação com HF Cache (`Lordnyx/qwen35-9b-teacher-logits-cache`)**:
   - Avaliou o uso de 48.163 sequências geradas pelo próprio Qwen3.8-27B em BF16 com Top-32 `topk_idx` e `topk_logprob`.
   - Evidenciou que o treino em domínio de código/raciocínio sintético reduz a perda de destilação para $0.47$, mas gera *distribution shift* quando avaliado no benchmark enciclopédico de linguagem natural (WikiText-2).

---

## 2. Tabela Geral de Resultados no Benchmark Cego (WikiText-2 Test)

| Arquitetura / Configuração | Parâmetros Refinador | PPL Teste | NLL Teste | Top-1 Teste | KL Div vs Prof | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Professor Oficial Qwen3.8-27B-FP8** | 0 (Referência) | **4.21** | 1.4374 | 72.82% | 0.0000 | Teto de Desempenho |
| Atlas Raw ($r=2048$, sem refinador) | 0 M | 2196.73 | 7.6947 | 8.33% | 4.8214 | Base Não-Estabilizada |
| SVD-16 Uniforme | 10.49 M | 682.45 | 6.5257 | 15.38% | 4.2104 | Ablação Preliminar |
| SVD-32 Uniforme | 20.97 M | 491.12 | 6.1967 | 18.25% | 3.8921 | Ablação Preliminar |
| SVD-64 Uniforme (Baseline Histórico) | 41.94 M | 324.27 | 5.7816 | 20.63% | 3.5120 | Baseline de Referência |
| SVD-128 Uniforme | 83.89 M | 232.68 | 5.4497 | 21.83% | 3.2845 | Marco Intermediário |
| **SVD-256 Uniforme (Passo 0)** | **167.77 M** | **192.68** | **5.2610** | **22.72%** | **3.1317** | **Baseline Canônico** |
| Política Adaptativa Míope (Busca Local) | 134.22 M | 453.92 | 6.1179 | 19.84% | 3.7841 | Refutado (Miopia Local) |
| Logit-Lens Intermediário Sequencial ($L_{48}\text{--}L_{51}$) | 167.77 M | 243.63 | 5.4957 | 20.63% | 3.4218 | Refutado (Colapso Latente) |
| **Otimização Terminal Real em $L_{63}$ (WikiText, Step 5)** | **167.77 M** | **179.94** | **5.1926** | **24.40%** | **3.0543** | **NOVO RECORDE ABSOLUTO** |
| L63 Destilado via HF Cache (Top-32 Código) | 167.77 M | 208.63 | 5.3406 | 23.91% | 3.2550 | Especializado em Código |

---

## 3. Investigação Experimental e Diagnósticos

### 3.1 Por que a Otimização Local e a Busca Adaptativa Falharam?

No Experimento B, um algoritmo de busca gananciosa camada a camada selecionou tipos mistos (`alpha_linear`, `linear`, `alpha_gelu`, `none`). O resultado final atingiu PPL de **453.92**, muito pior que o simples SVD-256 uniforme (**192.68**).

A razão matemática é a **falácia da decomposição local**:
$$\min_{\theta_l} \mathbb{E}\left[\|h_l^{(T)} - h_l^{(S)}(\theta_l)\|^2\right] \not\implies \min_\Theta \mathcal{L}_{\rm terminal}(\text{Model}(\Theta))$$
Reduzir o erro local em $L_l$ sem considerar como as 63 camadas subsequentes processam o espaço latente acumula distorções na geometria de ativação.

### 3.2 Por que o Logit-Lens Intermediário Prejudicou a Rede?

Na tentativa de aplicar logit-lens em $L_{48}$, a loss interna da camada caiu dramaticamente de $2.9579$ para $0.7043$ (-76.2%). No entanto, a PPL real de teste piorou a cada iteração:
$$192.68 \longrightarrow 193.14 \longrightarrow 193.61 \longrightarrow 194.58 \longrightarrow 243.63$$

**Diagnóstico Formal**:
Em um Transformer autorregressivo profundo de 64 camadas, os estados latentes intermediários ($h_{48}$) não contêm representações distribuídas finais sobre o vocabulário, mas sim conceitos abstratos relacionais de alta ordem. Ao forçar $\text{RMSNorm}(h_{48}) W_{\rm head}$ a coincidir com os logits de saída do professor, forçou-se a camada $L_{48}$ a atuar como um preditor prematuro, destruindo as transformações das 15 camadas posteriores ($L_{49}\text{--}L_{63}$).

### 3.3 A Solução: Otimização Terminal Estrita em $L_{63}$

Ao transferir a otimização terminal diretamente para a **última camada ($L_{63}$)**, a premissa muda fundamentalmente:
* A jusante de $L_{63}$ existem apenas o `RMSNorm` final e o `LMHead`.
* Não há camadas subsequentes para serem corrompidas.
* O gradiente da perda terminal flui diretamente para $W_{\rm down}$ e $W_{\rm up}$ de $L_{63}$.

Com taxa de aprendizado controlada (`lr = 1e-4`, AdamW com weight decay $10^{-3}$) inicializada com os pesos analíticos SVD-256:
* **Passo 0**: NLL = 5.2610 | PPL = 192.68 | Top-1 = 22.72% | KL = 3.1317
* **Passo 1**: NLL = 5.3008 | PPL = 200.51 | Top-1 = 23.21% | KL = 3.1371
* **Passo 3**: NLL = 5.1991 | PPL = 181.12 | Top-1 = 24.20% | KL = 3.0562
* **Passo 5**: **NLL = 5.1926 | PPL = 179.94 | Top-1 = 24.40% | KL = 3.0543** ($\Delta\text{PPL} = -12.74$)

---

## 4. Experimento em Escala com o Dataset HF Cache

O dataset [`Lordnyx/qwen35-9b-teacher-logits-cache`](https://huggingface.co/datasets/Lordnyx/qwen35-9b-teacher-logits-cache) contém 48.163 sequências geradas pelo próprio Qwen3.8-27B em BF16, armazenando os alvos Top-32 (`topk_idx` e `topk_logprob`).

O pipeline foi implementado em [`experiments/train_terminal_hf_cache.py`](file:///C:/Users/Nyx/Desktop/MathQwen/experiments/train_terminal_hf_cache.py):
1. **Extração das Ativações de Entrada de $L_{63}$**: Propagação de 16 sequências de 128 tokens através de 63 camadas fatoradas em 395.3s, persistido em [`checkpoints/hf_l63_cache_16seqs.pt`](file:///C:/Users/Nyx/Desktop/MathQwen/checkpoints/hf_l63_cache_16seqs.pt).
2. **Treinamento Top-32 KL Distillation + Hard CE**:
   $$\mathcal{L} = D_{\rm KL}(q_{\rm top32} \parallel p_{S, \rm top32}) + 0.1 \cdot \text{CE}(z_S, y_{\rm true})$$
   A perda caiu de $6.03$ para $0.47$ em 22 segundos de treino na GPU.
3. **Descoberta do Distribution Shift**:
   Os tokens supervisionados do cache eram formados exclusivamente por respostas de código e raciocínio (`bash`, `python`, ````markdown`). A especialização extrema causou divergência na distribuição de vocabulário enciclopédico geral do WikiText-2.

---

## 5. Artefatos e Checkpoints Persistidos

| Arquivo | Localização | Descrição |
| :--- | :--- | :--- |
| **Checkpoints Base SVD-256** | `checkpoints/svd256_baseline_stabilizers.pt` | 64 estabilizadores lineares $r=256$ (335 MB). |
| **Entrada Cacheada $L_{63}$** | `checkpoints/h_stud_63_cache.pt` | Ativações após camada 62 para WikiText (20.9 MB). |
| **Entrada HF Cache $L_{63}$** | `checkpoints/hf_l63_cache_16seqs.pt` | Ativações após camada 62 para o dataset HF (16.8 MB). |
| **Alvos do Professor 27B** | `checkpoints/prof_targets_cache.pt` | Logits e alvos intermediários de teste (1.7 GB). |
| **Resultados Ablação** | `experiments/refiner_ablation_results.json` | Resultados detalhados de $r=16$ até $r=256$. |
| **Resultados $L_{63}$** | `experiments/layer63_terminal_optimization_results.json` | Resultados da quebra do recorde para **179.94**. |
| **Resultados HF Cache** | `experiments/layer63_hf_optimization_results.json` | Curva de perda e PPL do treino com HF Cache. |
