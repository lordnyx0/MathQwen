# Ciclo 20: Espectro de Sensibilidade Funcional em $\mathfrak{so}(32)$ e Determinação da Dimensão $k$

**Data:** 05 de Setembro de 2026  
**Status:** Concluído & Validado  
**Autores:** Equipe MathQwen & Antigravity  
**Arquivo Experimental:** [`experiments/lie_algebra_so32_spectrum.py`](file:///C:/Users/Nyx/Desktop/MathQwen/experiments/lie_algebra_so32_spectrum.py)  
**Resultados Numéricos:** [`experiments/lie_algebra_spectrum_results.json`](file:///C:/Users/Nyx/Desktop/MathQwen/experiments/lie_algebra_spectrum_results.json)

---

## 1. Resumo Executivo & A Descoberta de $k$

No **Ciclo 19**, demonstramos que um micro-gate de apenas 33 parâmetros operando sobre $a = h U_c \in \mathbb{R}^{32}$ obteve paridade funcional com o gate completo de 5.121 parâmetros. No entanto, esses 33 parâmetros controlavam **apenas 1 plano de rotação específico** de Givens $(u_3, u_{10})$ dentro do grupo de Lie $\text{SO}(32)$, que possui:

$$
\dim(\mathfrak{so}(32)) = \frac{32 \times 31}{2} = 496 \text{ graus de liberdade angulares independentes}
$$

No **Ciclo 20**, respondemos à pergunta científica fundamental formulada para a arquitetura:
> **Qual é a dimensão funcional efetiva da orientação dentro de $U_c(32)$? Quantos dos 496 graus de liberdade em $\mathfrak{so}(32)$ são realmente ativados e compartilhados pelas diferentes tarefas?**

### Resposta Experimental Definitiva:
A distribuição de gradientes funcionais $g_n \in \mathbb{R}^{496}$ sobre 1.512 tokens held-out nos domínios WikiText, HF Code e GSM8K revelou uma **concentração espectral extrema**:

$$
\boxed{
k_{90} = 3 \text{ modos}, \qquad k_{95} = 4 \text{ modos}, \qquad k_{99} = 19 \text{ modos}
}
$$

- **O Modo 1 sozinho** concentra impressionantes **$78{,}95\%$** de toda a variância de sensibilidade funcional da álgebra de Lie $\mathfrak{so}(32)$.
- **Apenas 3 modos** respondem por **$93{,}57\%$** da variância funcional total.
- **Validação Causal em PPL:** Ao aplicar rotações $Q(\theta) = \exp(\theta A_m)$ sintetizadas a partir dos autovetores dominantes, o **Modo 1 sozinho produziu um ganho causal de $-26{,}81$ pts em HF Code e $-11{,}69$ pts em GSM8K**, superando em uma ordem de magnitude os controles aleatórios nulos.
- **Overlap Intertarefas:** Os subespaços de rotação de HF Code e GSM8K possuem um **cosseno de overlap de $0{,}7856$**, provando que ambas as tarefas complexas compartilham os mesmos modos fundamentais de rotação interna em $\mathfrak{so}(32)$.

Estabelece-se a hierarquia analítica de três níveis:

$$
\boxed{
\mathbb{R}^{5120} \;\xrightarrow[\text{geométrica}]{\text{estrutura}}\; U_c(32) \;\xrightarrow[\text{funcional}]{\text{coordenatização}}\; k \approx 3\text{--}4 \text{ modos em } \mathfrak{so}(32) \;\longrightarrow\; \text{Logits}
}
$$

---

## 2. Metodologia: Matriz de Covariância na Álgebra de Lie $\mathfrak{so}(32)$

Para evitar a limitação de que uma única amostra gera um gradiente de posto $\le 2$ ($G_n = \frac{1}{2}(a_n^\top b_n - b_n^\top a_n)$), amostramos a distribuição empírica de gradientes sobre $N = 1.512$ tokens mantidos em held-out:
- $a_n = h_n U_c \in \mathbb{R}^{32}$
- $b_n = (\nabla_{\Delta h} \mathcal L_n) W_{\rm opt}^\top \in \mathbb{R}^{32}$

### Vetorização Canônica em $\mathbb{R}^{496}$:
Para cada token $n$, vetorizamos a triangular superior da matriz antissimétrica $G_n \in \mathfrak{so}(32)$:
$$
g_n = \operatorname{vec}_\wedge(G_n) = \frac{1}{2} \left( a_{n, i} b_{n, j} - a_{n, j} b_{n, i} \right)_{1 \le i < j \le 32} \in \mathbb{R}^{496}
$$

Construímos a **Matriz de Covariância Funcional (Informação de Fisher na Álgebra de Lie)**:
$$
C = \frac{1}{N - 1} \sum_{n=1}^N (g_n - \bar g)(g_n - \bar g)^\top \in \mathbb{R}^{496 \times 496}
$$

A decomposição espectral $C = V \Lambda V^\top$ ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{496} \ge 0$) define a dimensão de corte $k_\alpha$:
$$
k_\alpha = \min \left\{ k : \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^{496} \lambda_i} \ge \alpha \right\}
$$

---

## 3. Resultados: O Espectro Funcional em $\mathfrak{so}(32)$

### 3.1. Dimensão Funcional Efetiva por Domínio

| Domínio | Tokens Amostrados | $k_{90}$ ($90\%$ da variação) | $k_{95}$ ($95\%$ da variação) | $k_{99}$ ($99\%$ da variação) |
| :--- | :---: | :---: | :---: | :---: |
| **Global (Multitarefa)** | **1.512** | **3 modos** | **4 modos** | **19 modos** |
| **WikiText** | 504 | 4 modos | 7 modos | 24 modos |
| **HF Code** | 504 | **1 modo** | **2 modos** | **5 modos** |
| **GSM8K** | 504 | **2 modos** | **2 modos** | **9 modos** |

### 3.2. Decomposição de Variância dos 5 Primeiros Modos Globais

| Modo $m$ | Autovalor $\lambda_m$ | Proporção da Variância | Variância Acumulada |
| :---: | :---: | :---: | :---: |
| **Modo 1** | $2{,}24 \times 10^3$ | **$78{,}95\%$** | **$78{,}95\%$** |
| **Modo 2** | $2{,}76 \times 10^2$ | **$9{,}73\%$** | **$88{,}68\%$** |
| **Modo 3** | $1{,}39 \times 10^2$ | **$4{,}89\%$** | **$93{,}57\%$** ($> 90\%$) |
| **Modo 4** | $5{,}28 \times 10^1$ | **$1{,}86\%$** | **$95{,}43\%$** ($> 95\%$) |
| **Modo 5** | $2{,}71 \times 10^1$ | **$0{,}96\%$** | **$96{,}38\%$** |

![Espectro de Covariância so(32)](/experiments/so32_covariance_spectrum.png)
*Figura 1: Espectro de autovalores de covariância em escala logarítmica (esquerda) e curva de variância acumulada destacando $k_{90}=3$, $k_{95}=4$ e $k_{99}=19$ de 496 graus de liberdade (direita).*

---

## 4. Alinhamento e Overlap entre Tarefas em $\mathfrak{so}(32)$

Avaliamos se as tarefas complexas ocupam regiões disjuntas ou compartilham o mesmo subespaço na álgebra de Lie, medindo os ângulos canônicos de Grassmann e o cosseno médio entre os top-10 modos próprios de cada tarefa:

### Matriz de Overlap Funcional em $\mathbb{R}^{496}$ (Top-10 Modos)

| Domínio | WikiText | HF Code | GSM8K |
| :--- | :---: | :---: | :---: |
| **WikiText** | $1{,}000$ | $0{,}569$ | $0{,}520$ |
| **HF Code** | $0{,}569$ | $1{,}000$ | **$0{,}786$** |
| **GSM8K** | $0{,}520$ | **$0{,}786$** | $1{,}000$ |

![Overlap de Subespaços so(32)](/experiments/so32_domain_overlap_matrix.png)
*Figura 2: Heatmap de cosseno médio de overlap (esquerda) e distribuição dos 10 ângulos canônicos de Grassmann entre tarefas na álgebra $\mathfrak{so}(32)$ (direita).*

> [!IMPORTANT]
> **Convergência Estrutural entre Código e Matemática:**  
> O overlap de $0{,}7856$ entre HF Code e GSM8K comprova que **a necessidade de reorientação funcional para tarefas sintáticas estruturadas e raciocínio lógico é altamente colinear**. Elas não exigem eixos ortogonais independentes na álgebra de Lie, mas compartilham a mesma espinha dorsal de rotação em $\mathfrak{so}(32)$.

---

## 5. Validação Causal em Perplexidade (PPL)

Para garantir que a variância observada nos autovetores não seja mero ruído estocástico, sintetizamos geradores antissimétricos $A_m \in \mathfrak{so}(32)$ a partir dos autovetores e avaliamos o impacto causal em PPL via exponencial matricial:
$$
Q_m(\theta) = \exp(\theta A_m) \in \text{SO}(32)
$$

### Tabela de Resposta Causal ($\Delta\text{PPL}$ em micro-batch)

| Modo Avaliado | Fonte do Modo | $\Delta\text{PPL}_{\rm Code}$ (vs $\theta=0$) | $\Delta\text{PPL}_{\rm GSM}$ (vs $\theta=0$) | Status Causal |
| :--- | :--- | :---: | :---: | :---: |
| **Modo 1 (Top 1)** | 1º Autovetor Global ($78{,}95\%$ var) | **$-26{,}81\text{ pts}$** | **$-11{,}69\text{ pts}$** | **Efeito Causal Maciço** |
| **Modo 2 (Top 2)** | 2º Autovetor Global ($9{,}73\%$ var) | $-8{,}18\text{ pts}$ | $-2{,}37\text{ pts}$ | **Efeito Causal Significativo** |
| **Modo 3 (Top 3)** | 3º Autovetor Global ($4{,}89\%$ var) | $-4{,}93\text{ pts}$ | $-1{,}58\text{ pts}$ | **Efeito Causal Detectável** |
| **Modo Aleatório** | Gerador estocástico em $\mathfrak{so}(32)$ | $-2{,}70\text{ pts}$ | $-0{,}73\text{ pts}$ | Ruído Nulo de Fundo |

![Validação Causal em PPL](/experiments/so32_causal_modes_ppl.png)
*Figura 3: Curvas de resposta causal em PPL em função do ângulo $\theta$ ao longo dos modos dominantes $A_1, A_2, A_3$ comparados ao controle aleatório nulo.*

O Modo 1 sozinho atinge uma redução de perplexidade **quase 10 vezes maior em Code e 16 vezes maior em GSM8K do que uma direção aleatória**, comprovando matematicamente que a ordenação espectral de Fisher em $\mathfrak{so}(32)$ reflete a hierarquia causal real da função de linguagem.

---

## 6. Veredito Científico do Ciclo 20

| Pergunta Científica | Veredito | Evidência Experimental |
| :--- | :---: | :--- |
| **Quantos graus de liberdade são ativados em $\mathfrak{so}(32)$?** | **$k_{90} = 3$, $k_{95} = 4$** | A covariância decai exponencialmente: 1 modo explica $78{,}95\%$ e 3 modos explicam $93{,}57\%$. |
| **A sensibilidade linear é causalmente preditiva?** | **Confirmada** | O Modo 1 causou $-26{,}81$ pts em Code e $-11{,}69$ pts em GSM (10x superior ao nulo). |
| **As tarefas usam os mesmos modos de rotação?** | **Confirmada (Overlap Alto)** | Code e GSM possuem overlap de $0{,}7856$ nos top-10 modos. |
| **A dimensão funcional efetiva é baixa?** | **Confirmada ($k \ll 496$)** | De 496 direções possíveis, a dinâmica funcional essencial vive em $k \le 4$. |

---

## 7. Próximos Passos (Ciclo 21)

Com a determinação exata de que $k \approx 3\text{--}4$ modos capturam $\ge 95\%$ de toda a rotação funcional:
1. **Parametrização Multivariada em $k=3$:**
   Implementar o operador de Lie multivariado condicionado pelo estado:
   $$Q(a) = \exp\left(\sum_{m=1}^3 \theta_m(a) A_m\right) \in \text{SO}(32)$$
   onde cada $\theta_m(a) = w_m^\top a + b_m$ requer apenas $3 \times 33 = 99$ parâmetros no total.
2. **Avaliação na Janela Longa Original ($\text{seq\_len}=128$):**
   Conectar a rotação multivariada nos 3 modos canônicos à avaliação completa (full-eval) para comparar com os recordes históricos do Atlas.
