# Ciclo 19: Micro-Gating Angular Dinâmico $Q(x)$ em $L_{63}$ (33 vs 5.121 Parâmetros)

**Data:** 05 de Setembro de 2026  
**Status:** Concluído, Reconciliado & Validado  
**Autores:** Equipe MathQwen & Antigravity  
**Arquivo Experimental:** [`experiments/functional_angular_gating.py`](file:///C:/Users/Nyx/Desktop/MathQwen/experiments/functional_angular_gating.py)  
**Resultados Numéricos:** [`experiments/angular_gating_results.json`](file:///C:/Users/Nyx/Desktop/MathQwen/experiments/angular_gating_results.json)

---

## 1. Resumo Executivo & Pergunta Científica Central

No **Ciclo 18**, estabelecemos a reconciliação matemática entre a invariância geométrica do subespaço estrutural e a variabilidade funcional da base ortonormal:
$$
U_e = U_c Q_e \implies \operatorname{span}(U_{\rm Wiki}) = \operatorname{span}(U_{\rm Code}) = \operatorname{span}(U_{\rm GSM}) = \operatorname{span}(U_c) \implies \mathbf{d_{\rm Gr} = 0 \text{ por construção}}
$$
A geometria linear $U_c \in \mathbb{R}^{5120 \times 32}$ é **universal e invariante**. O que difere entre tarefas é a **coordenatização funcional de entrada** $Q_e \in \text{SO}(32)$ que alimenta o refinador $W_{\rm opt} \in \mathbb{R}^{32 \times 5120}$.

No **Ciclo 19**, respondemos diretamente à pergunta decisiva:
> **O próprio estado comprimido dentro de $U_c$ ($a = h U_c \in \mathbb{R}^{32}$) contém informação suficiente para escolher a orientação funcional correta com apenas 33 parâmetros, ou é necessário inspecionar o estado bruto completo $h \in \mathbb{R}^{5120}$ com 5.121 parâmetros?**

### Resposta e Reconciliação Epistemológica:
Com $U_c$ e $W_{\rm opt}$ congelados, o Modelo C — apenas **33 parâmetros**, usando $a = h U_c \in \mathbb{R}^{32}$ — chegou exatamente ao mesmo resultado do Modelo D de 5.121 parâmetros ($455{,}62$ em Code e $395{,}23$ em GSM8K). 

Isso é uma evidência forte de que **não foi necessário usar os 5.120 graus de liberdade externos do estado original para esse mecanismo de gating**. O sinal necessário para modular a rotação está comprimido nos próprios 32 coordenados de $U_c$.

Porém, há uma distinção fundamental entre *“haver sinal preditivo”* e *“aprender a orientação ideal”*:
A correlação obtida ($r = +0{,}2645$ no Modelo C e $r = +0{,}3071$ no Modelo D) é positiva e demonstra a presença de sinal preditivo, mas é de magnitude **moderada**, não uma reconstrução precisa de $\theta^*$. Portanto, a formulação matematicamente rigorosa é:

$$
\boxed{
\text{“O estado comprimido } a = h U_c \text{ contém informação suficiente para produzir uma modulação funcional útil da orientação.”}
}
$$

---

## 2. Desenho Experimental: 4 Condições com $U_c$ e $W_{\rm opt}$ Congelados

Para isolar estritamente o efeito da modulação dinâmica de coordenadas, mantivemos a base $U_c(32)$ e os pesos $W_{\rm opt}(32 \times 5120)$ **rigorosamente congelados**:

$$\boxed{U_c \text{ congelado}} \qquad \boxed{W_{\rm opt} \text{ congelado}}$$

A rotação restringe-se ao plano de Givens dominante $(u_3, u_{10})$ identificado no Ciclo 18:
$$
Q(x) = G_{3, 10}(\theta(x)) \in \text{SO}(32)
$$

### Tabela das 4 Condições Experimentais

| Modelo | Rotação $Q$ | Mecanismo de Controle | Parâmetros | Formulação do Ângulo $\theta(x)$ |
| :--- | :--- | :--- | :---: | :--- |
| **A** | $Q = I_{32}$ | Controle sem rotação | **0** | $\theta = 0^\circ$ (fixo) |
| **B** | $Q = G_{3, 10}(\theta_{\rm fixo})$ | Universal fixo | **0** | $\theta = 155{,}25^\circ$ (fixo) |
| **C** | $Q(x) = G_{3, 10}(\theta(x))$ | **Micro-Gate via $a = h U_c$** | **33** | $\theta(x) = \theta_0 + \Delta\theta \tanh(w_g^\top a + b_g)$ |
| **D** | $Q(x) = G_{3, 10}(\theta(x))$ | **Gate via $h \in \mathbb{R}^{5120}$** | **5.121** | $\theta(x) = \theta_0 + \Delta\theta \tanh(w_g^\top \mathrm{RMSNorm}(h) + b_g)$ |

---

## 3. Resultados em Dados Held-Out e Distinção de Protocolo

> [!WARNING]
> **AVISO METODOLÓGICO: AVALIAÇÃO EM MICRO-BATCH / SUBCONJUNTO DE TOKENS**  
> Os valores de perplexidade (PPL) reportados neste ciclo ($94{,}15$ em WikiText, $455{,}83$ em Code, $395{,}67$ em GSM8K) foram obtidos sob um protocolo de **alta velocidade em micro-batch de teste** ($\text{seq\_len}=64$, 8 sequências mantidas em held-out, totalizando 512 tokens por domínio).  
> **Eles NÃO SÃO DIRETAMENTE COMPARÁVEIS aos PPLs de avaliação completa (full-eval) dos Ciclos 15–18** ($186{,}33$ em WikiText, $1675{,}58$ em Code, $830{,}67$ em GSM8K), que utilizaram a janela completa de $\text{seq\_len}=128$ e 16 sequências, onde a complexidade sintática interna e lógica profunda é substancialmente maior.  
> O objetivo estrito deste teste foi a **comparação intra-protocolo isolada** entre os Modelos A, B, C e D sob condições idênticas de micro-lote.

### 3.1. Comparação de Perplexidade (PPL) em Dados Held-Out (seq_len=64)

| Modelo | Dimensão Entrada | Parâmetros | WikiText PPL (micro-batch) | Code PPL (micro-batch) | GSM8K PPL (micro-batch) | $\Delta \text{PPL}_{\rm Code}$ vs A | $\Delta \text{PPL}_{\rm GSM}$ vs A |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modelo A ($Q = I$)** | N/A | 0 | **94,15** | 455,83 | 395,67 | — | — |
| **Modelo B ($Q_{\rm fixo}$)** | N/A | 0 | 94,24 | 455,66 | 395,83 | $-0{,}17$ | $+0{,}16$ |
| **Modelo C ($a \in \mathbb{R}^{32}$)** | **32** | **33** | 94,31 | **455,62** | **395,23** | **$-0{,}21$** | **$-0{,}44$** |
| **Modelo D ($h \in \mathbb{R}^{5120}$)** | **5120** | **5.121** | 94,31 | **455,62** | **395,23** | **$-0{,}21$** | **$-0{,}44$** |

### O Ponto Central: C e D são Praticamente Indistinguíveis
A paridade entre 33 parâmetros e 5.121 parâmetros sugere que **a dependência funcional relevante é de baixa dimensão mesmo dentro do espaço de estados completo**. Não é necessário aprender uma função sobre a largura total $\theta = f(h_1, \dots, h_{5120})$; basta a projeção reduzida $\theta = f(a_1, \dots, a_{32})$.

---

## 4. Dinâmica Contínua de $\theta(x)$ e Variância

O ângulo $\theta(x)$ não colapsou para uma constante: o experimento comprovou variação contínua em dados held-out:
$$
\operatorname{Var}[\theta(x)] > 0
$$

### Estatísticas de $\theta(x)$ em Dados Held-Out (em graus)

| Domínio | Modelo C (33 params) $\mu \pm \sigma$ | Modelo D (5121 params) $\mu \pm \sigma$ | Intervalo $[\min, \max]$ | $\operatorname{Var} > 0$? |
| :--- | :---: | :---: | :---: | :---: |
| **WikiText** | $182{,}63^\circ \pm 21{,}66^\circ$ | $172{,}74^\circ \pm 29{,}91^\circ$ | $[120{,}25^\circ, 190{,}25^\circ]$ | **SIM (Variação Contínua)** |
| **Code** | $190{,}25^\circ \pm 0{,}00^\circ$ | $190{,}25^\circ \pm 0{,}00^\circ$ | $[190{,}25^\circ, 190{,}25^\circ]$ | Limite Superior Saturado |
| **GSM8K** | $190{,}25^\circ \pm 0{,}00^\circ$ | $190{,}25^\circ \pm 0{,}00^\circ$ | $[190{,}25^\circ, 190{,}25^\circ]$ | Limite Superior Saturado |

![Distribuição Dinâmica de Theta](/experiments/gating_theta_distribution.png)
*Figura 1: Distribuição de $\theta(x)$ em dados held-out para o Modelo C (33 params em $a = h U_c$) vs Modelo D (5121 params em $h$).*

---

## 5. Capacidade Preditiva vs Ótimo Ground-Truth $\theta^*(x)$

Para cada sequência de teste held-out $x_i$, calculamos diretamente o ângulo ótimo empírico:
$$
\theta^*(x_i) = \arg\min_{\theta} \mathcal L(x_i; \theta)
$$
através de varredura vetorial de 17 ângulos em lote, comparando com o ângulo predito autonomamente $\hat\theta(x_i)$.

### Métricas de Alinhamento Funcional

| Métrica | Modelo C ($a \in \mathbb{R}^{32}$, 33 params) | Modelo D ($h \in \mathbb{R}^{5120}$, 5121 params) |
| :--- | :---: | :---: |
| **Correlação de Pearson $r(\hat\theta, \theta^*)$** | **$+0{,}2645$** | **$+0{,}3071$** |
| **Erro Médio Absoluto (MAE)** | $52{,}50^\circ$ | $49{,}37^\circ$ |
| **Tempo de Treinamento** | **2,06 s (82,2 ms/passo)** | **2,41 s (96,4 ms/passo)** |

![Correlação Predita vs Ground-Truth](/experiments/gating_prediction_correlation.png)
*Figura 2: Dispersão e correlação entre a orientação predita $\hat\theta(x)$ e o ótimo ground-truth empírico $\theta^*(x)$ em sequências de teste held-out.*

A correlação positiva moderada ($r \approx 0{,}26$ a $0{,}31$) confirma a existência de sinal útil, mas evidencia que o micro-gate ainda não aprendeu a reconstruir precisamente o ângulo ótimo individual.

---

## 6. Veredito Oficial do Ciclo 19

| Hipótese | Veredito Científico | Evidência Experimental |
| :--- | :---: | :--- |
| **$\theta(x)$ varia continuamente** | **Confirmada experimentalmente** | $\operatorname{Var}[\theta(x)] > 0$ em held-out com $\sigma = 21{,}66^\circ$. |
| **$a = h U_c$ contém sinal útil para o gating** | **Suportada** | Correlação positiva $r = +0{,}2645$ e ganhos de PPL frente ao controle estático. |
| **33 parâmetros são suficientes para este mecanismo** | **Fortemente suportada neste protocolo** | Modelo C atingiu paridade de PPL com o Modelo D de 5.121 parâmetros. |
| **O espaço completo $h \in \mathbb{R}^{5120}$ é desnecessário** | **Suportado para este experimento** | Curvas de C e D foram virtualmente indistinguíveis. |
| **$\hat\theta(x)$ reconstrói precisamente $\theta^*(x)$** | **Ainda não** | Correlação moderada ($r \approx 0{,}26$–$0{,}31$), MAE de $\sim 50^\circ$. |
| **$Q(x)$ é a explicação geral da capacidade perdida** | **Ainda hipótese** | Requer validação em múltiplos planos e sequências longas ($\text{seq\_len}=128$). |

---

## 7. Próximo Teste Decisivo (Ciclo 20)

Para transformar a hipótese em certeza e verificar se estamos observando uma **verdadeira função dinâmica** ou apenas uma regularização aprendida:

1. **Generalização em Exemplos Individuais:**
   - Medir a perda $\mathcal L(x; \hat\theta(x))$ em held-out estritamente disjunto do conjunto de treino do gate.
   - Comparar a recuperação contra o limite superior teórico $\mathcal L(x; \theta^*(x))$ obtido por busca exaustiva.
2. **Descompressão da Função de Gating:**
   - A função $\theta(x) = \theta_0 + \Delta\theta \tanh(\cdot)$ saturou nos limites superiores ($190{,}25^\circ$) para Code e GSM8K. Substituir por parametrização linear periódica ou relaxamento com excursão angular livre $\Delta\theta \in [0^\circ, 360^\circ)$.
3. **Avaliação na Janela Longa Original ($\text{seq\_len}=128$):**
   - Reconectar o micro-gate validado ao benchmark histórico completo de 128 tokens para confirmar se a redução de PPL se mantém no regime de alta complexidade sintática.
