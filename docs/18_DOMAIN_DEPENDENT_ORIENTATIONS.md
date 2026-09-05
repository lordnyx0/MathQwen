# 18. Mapeamento de Coordenatizações Funcionais em SO(32) dentro da Variedade Compartilhada Uc em L63

**Repositório:** `lordnyx0/MathQwen`  
**Alvo Experimental:** Camada Terminal 63 ($L_{63}$)  
**Data:** 05 de Setembro de 2026  
**Status da Hipótese:** $\boxed{\mathbf{CONFIRMADA\ (A\ ESTRADA\ COMPARTILHADA\ E\ AS\ COORDENADAS\ POR\ DOMÍNIO)}}$  
*(Comprovou-se empiricamente que o subespaço $U_c(32)$ forma a variedade estrutural invariante — $d_{\rm Gr} = 0$ por construção —, enquanto a orientação/coordenatização interna $Q_e \in \text{SO}(32)$ é especializada por domínio. A geometria estrutural permanece fixa; a função escolhe coordenadas diferentes dentro dela).*

---

## Veredito Científico Consolidado

A partir da revisão matemática rigorosa da formulação deste ciclo, o estado científico é formalmente classificado da seguinte forma:

| Afirmação Científica | Veredito | Evidência Experimental / Fundamentação |
| :--- | :---: | :--- |
| **$U_c$ é compartilhado universalmente** | **Fortemente Suportado** | $\operatorname{span}(U_{\rm Wiki}) = \operatorname{span}(U_{\rm Code}) = \operatorname{span}(U_{\rm GSM}) = \operatorname{span}(U_c)$. O subespaço de 32 canais ancora todas as distribuições. |
| **Existem orientações angulares $\theta^*$ diferentes por domínio** | **Suportado** | $\theta^*_{\rm Wiki} = 155{,}25^\circ$, $\theta^*_{\rm Code} = 131{,}50^\circ$, $\theta^*_{\rm GSM} = 163{,}75^\circ$ no plano canônico $(u_3, u_{10})$. |
| **Existe especialização funcional de coordenadas em $U_c$** | **Promissor / Suportado** | Cada distribuição obtém seu melhor desempenho sob sua própria rotação $Q_e$ (a diagonal da matriz de transferência cruzada é o mínimo de cada linha). |
| **As diferenças entre $U_e$ são distâncias de Grassmann** | **❌ Incorreto** | Como $Q_e \in \text{SO}(32)$ é ortogonal, $\operatorname{span}(U_e) = \operatorname{span}(U_c)$. Logo, $\mathbf{d_{\rm Gr}(U_A, U_B) = 0}$ identicamente por construção. |
| **$\Delta\theta^* = 8{,}50^\circ, 23{,}75^\circ, 32{,}25^\circ$ medem separação no Grassmanniano** | **❌ Incorreto** | Trata-se de **distâncias angulares geodésicas no grupo de Lie $\text{SO}(32)$** (especificamente no subgrupo $\text{SO}(2)$ do plano Givens), medindo rotação de *frames* dentro do mesmo subespaço. |
| **Um micro-gating $Q(x)$ pode selecionar a orientação a partir do estado** | **Hipótese Muito Plausível** | Próxima etapa do programa: parametrizar a rotação dinamicamente a partir do estado oculto $h(x)$ sem alterar o espaço base $U_c$. |

---

## 1. Correção Matemática Fundamental: Grassmanniano vs. Grupo de Lie SO(32)

### A Distinção Canônica

O espaço de refinamento terminal é definido pela aplicação:
$$\Delta h = (h U_e) W_{\rm opt} = (h U_c Q_e) W_{\rm opt}, \qquad Q_e \in \text{SO}(32)$$

onde $U_c \in \mathbb{R}^{5120 \times 32}$ com $U_c^T U_c = I_{32}$, e $W_{\rm opt} \in \mathbb{R}^{32 \times 5120}$ é o refinador linear fixo.

Como $Q_e$ é uma matriz ortogonal de posto completo ($32 \times 32$):
$$\operatorname{span}(U_e) = \operatorname{span}(U_c Q_e) = \operatorname{span}(U_c) \subset \mathbb{R}^{5120}$$

Por conseguinte:
$$\boxed{U_{\rm Wiki}, \ U_{\rm Code}, \ U_{\rm GSM} \text{ são exatamente o mesmo ponto na variedade Grassmanniana } \text{Gr}(32, 5120)}$$

Para qualquer par de domínios $A$ e $B$:
$$M = U_A^T U_B = Q_A^T U_c^T U_c Q_B = Q_A^T Q_B$$

Como o produto de matrizes ortogonais é estritamente ortogonal, todos os seus valores singulares são unitários:
$$\sigma_k(U_A^T U_B) = 1 \quad (\forall k \in \{1, \dots, 32\}) \implies \theta_k = \arccos(1) = 0$$

Portanto:
$$\boxed{d_{\rm Gr}(U_A, U_B) = 0 \quad (\forall A, B)}$$

### O que as defasagens $\Delta\theta^*$ realmente medem?

O Grassmanniano $\text{Gr}(k, d)$ é o espaço de todos os subespaços lineares de dimensão $k$, sendo invariante sob qualquer mudança de base interna:
$$\text{Gr}(k, d) \cong \text{O}(d) / (\text{O}(k) \times \text{O}(d-k))$$

Quando variamos a matriz de Givens $G_{ij}(\theta) \in \text{SO}(32)$, **não estamos nos movendo no Grassmanniano**. Estamos nos movendo ao longo de uma geodésica no **grupo de Lie $\text{SO}(32)$**:

$$\boxed{
\text{Grassmanniano mede o subespaço: } d_{\rm Gr} = 0
}
\qquad
\boxed{
\text{SO}(32) \text{ mede a coordenatização/orientação funcional dentro dele: } d_{\text{SO}(32)} = \sqrt{2}|\theta_A^* - \theta_B^*|
}$$

As defasagens encontradas:
- WikiText vs GSM8K: $\Delta \theta^* = 8{,}50^\circ \implies d_{\text{SO}(32)} = 0{,}2098\text{ rad}$
- WikiText vs HF Code: $\Delta \theta^* = 23{,}75^\circ \implies d_{\text{SO}(32)} = 0{,}5862\text{ rad}$
- HF Code vs GSM8K: $\Delta \theta^* = 32{,}25^\circ \implies d_{\text{SO}(32)} = 0{,}7959\text{ rad}$

representam o **ângulo de rotação dos eixos funcionais** que alimentam o mapeamento linear $W_{\rm opt}$.

---

## 2. A Estrada Compartilhada e as Coordenadas por Domínio

Essa distinção matemática reforça e refina a intuição física do projeto:

$$
\boxed{
\text{Mesmo Subespaço} \neq \text{Mesma Base / Coordenatização Funcional}
}
$$

1. **A Estrada:** O espaço linear gerado por $U_c(32)$ é comum a todos os domínios. Nenhuma direção extra precisa ser buscada fora dele.
2. **A Faixa:** A projeção de subida $W_{\rm opt}$ foi calibrada sob um determinado referencial. Mudar a orientação $Q_e$ atua como um ajuste de sintonia fina nas coordenadas de entrada de $W_{\rm opt}$, redirecionando o ganho para os modos mais relevantes de cada distribuição.

---

## 3. Resultados Experimentais da Varredura e Refinamento

### Tabela 1: Resumo da Varredura 360° em 8 Planos Canônicos de Givens (Passo de 10°)

| Plano Canônico | Par de Modos | Descrição Funcional | WikiText $\theta^*$ (PPL) | HF Code $\theta^*$ (PPL) | GSM8K $\theta^*$ (PPL) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Plano 1** | $(u_4, u_{10})$ | Top 1 e Top 2 de Alavancagem | $50^\circ$ ($186{,}58$) | $130^\circ$ ($1678{,}07$) | $90^\circ$ ($832{,}70$) |
| **Plano 2** | $(u_4, u_5)$ | Top 1 e Top 3 de Alavancagem | $50^\circ$ ($186{,}55$) | $130^\circ$ ($1679{,}76$) | $90^\circ$ ($833{,}43$) |
| **Plano 3** | $(u_{10}, u_5)$ | Top 2 e Top 3 de Alavancagem | $160^\circ$ ($186{,}48$) | $190^\circ$ ($1679{,}36$) | $200^\circ$ ($832{,}12$) |
| **Plano 4** | $(u_3, u_4)$ | Top 4 e Top 1 de Alavancagem | $280^\circ$ ($186{,}51$) | $180^\circ$ ($1678{,}37$) | $240^\circ$ ($832{,}15$) |
| **Plano 5** | $\mathbf{(u_3, u_{10})}$ | **Top 4 e Top 2 (Plano Dominante)** | $\mathbf{150^\circ}$ ($\mathbf{186{,}37}$) | $\mathbf{130^\circ}$ ($\mathbf{1675{,}83}$) | $\mathbf{160^\circ}$ ($\mathbf{830{,}77}$) |
| **Plano 6** | $(u_5, u_7)$ | Top 3 e Top 5 de Alavancagem | $110^\circ$ ($186{,}51$) | $180^\circ$ ($1677{,}84$) | $200^\circ$ ($831{,}21$) |
| **Plano 7** | $(u_1, u_4)$ | Energia Máxima vs Alavancagem | $20^\circ$ ($186{,}56$) | $50^\circ$ ($1685{,}59$) | $0^\circ$ ($834{,}74$) |
| **Plano 8** | $(u_1, u_2)$ | Modos Dominantes do Erro Euclidiano | $10^\circ$ ($186{,}61$) | $60^\circ$ ($1680{,}52$) | $40^\circ$ ($833{,}94$) |

### Tabela 2: Refinamento Hierárquico de Alta Precisão ($0{,}25^\circ$) nos Planos Prioritários

| Domínio | Plano | Ângulo Ótimo $\theta^*$ | PPL Refinada | Delta vs Atlas Raw | Delta vs $\theta=0$ | Top-1 Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **WikiText** | $(u_3, u_{10})$ | $\mathbf{155{,}25^\circ}$ | $\mathbf{186{,}33}$ | $\mathbf{-5{,}04\text{ pts}}$ | $-0{,}34\text{ pts}$ | $22{,}72\%$ |
| WikiText | $(u_{10}, u_5)$ | $154{,}00^\circ$ | $186{,}45$ | $-4{,}91\text{ pts}$ | $-0{,}22\text{ pts}$ | $22{,}82\%$ |
| **HF Code** | $(u_3, u_{10})$ | $\mathbf{131{,}50^\circ}$ | $\mathbf{1675{,}58}$ | $+26{,}22\text{ pts}$ | $\mathbf{-14{,}45\text{ pts}}$ | $9{,}89\%$ |
| HF Code | $(u_{10}, u_5)$ | $191{,}00^\circ$ | $1679{,}31$ | $+29{,}95\text{ pts}$ | $-10{,}72\text{ pts}$ | $10{,}09\%$ |
| **GSM8K** | $(u_3, u_{10})$ | $\mathbf{163{,}75^\circ}$ | $\mathbf{830{,}67}$ | $\mathbf{-12{,}39\text{ pts}}$ | $\mathbf{-4{,}07\text{ pts}}$ | $\mathbf{10{,}97\%}$ |
| GSM8K | $(u_{10}, u_5)$ | $198{,}00^\circ$ | $832{,}09$ | $-10{,}96\text{ pts}$ | $-2{,}65\text{ pts}$ | $10{,}97\%$ |

---

## 4. Análise Realista da Matriz de Transferência Cruzada

A matriz de transferência mede a perplexidade resultante ao avaliar os dados do domínio $Y$ sob a rotação $Q_X$:

### Tabela 3: Matriz de Transferência Cruzada $3 \times 3$ (Perplexidade PPL)

| Avaliação no Dado | Usando $Q_{\rm Wiki}$ ($\theta=155{,}25^\circ$) | Usando $Q_{\rm Code}$ ($\theta=131{,}50^\circ$) | Usando $Q_{\rm GSM}$ ($\theta=163{,}75^\circ$) | Atlas Raw (Sem Refinador) | Refinador Base ($\theta=0^\circ$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WikiText (Test)** | $\mathbf{186{,}33^*}$ | $186{,}52$ | $186{,}45$ | $191{,}37$ | $186{,}67$ |
| **HF Code** | $1676{,}20$ | $\mathbf{1675{,}58^*}$ | $1676{,}59$ | $1649{,}36$ | $1690{,}03$ |
| **GSM8K** | $830{,}76$ | $830{,}90$ | $\mathbf{830{,}67^*}$ | $843{,}06$ | $834{,}74$ |

*(\* indica o ótimo do próprio domínio)*

### Interpretação Moderada e Precisa dos Números

Embora a diagonal seja o ponto ótimo em todas as linhas, a magnitude das diferenças fora da diagonal exige uma leitura científica sóbria:

1. **WikiText vs. GSM8K (Proximidade Quase Perfeita):**
   - No WikiText, usar $Q_{\rm GSM}$ em vez de $Q_{\rm Wiki}$ custa apenas **$+0{,}12\text{ pts}$** de PPL ($186{,}45$ vs $186{,}33$).
   - No GSM8K, usar $Q_{\rm Wiki}$ em vez de $Q_{\rm GSM}$ custa meros **$+0{,}09\text{ pts}$** de PPL ($830{,}76$ vs $830{,}67$).
   - Ambas as orientações continuam muito superiores ao ponto base $\theta=0$ ($186{,}67$ e $834{,}74$). Ou seja, **linguagem natural e matemática compartilham uma orientação quase idêntica**.
2. **HF Code vs. Linguagem:**
   - No Code, usar $Q_{\rm Code}$ dá $1675{,}58$, enquanto $Q_{\rm Wiki}$ dá $1676{,}20$. A diferença entre as rotações é de **$0{,}62\text{ pt}$**.
   - Isso mostra que a rotação específica melhora o resultado, mas não há um colapso catastrófico ao cruzar as orientações.
3. **Conclusão sobre "Especialização":**
   Existe especialização funcional mensurável ($Q_{\rm Code} \ne Q_{\rm Wiki} \ne Q_{\rm GSM}$), mas não uma "separação profunda ou ortogonal". As três faixas são pequenas modulações angulares dentro da mesma estrada compartilhada.

---

## 5. Próximo Ciclo de Engenharia: De $Q_e$ Estático a $Q(x)$ Dinâmico

O encerramento formal deste ciclo estabelece a tese definitiva:

$$\boxed{
\textbf{A geometria estrutural permanece fixa; a função escolhe coordenadas diferentes dentro dela.}
}
$$

Portanto, o próximo passo de engenharia não deve criar novos subespaços nem aumentar o rank. O objetivo é unificar as rotações estáticas por domínio:

$$\boxed{
Q_{\rm Wiki}, \ Q_{\rm Code}, \ Q_{\rm GSM} \quad \longrightarrow \quad Q(x)
}$$

### Proposta Arquitetural: Micro-Gating Angular em SO(2)

Como a principal rotação funcional ocorre no plano canônico $(u_3, u_{10})$, podemos parametrizar a orientação dinamicamente a partir do estado oculto de entrada $h \in \mathbb{R}^{5120}$:

$$\theta(x) = \theta_0 + \Delta \theta \cdot \tanh(w_g^T \text{RMSNorm}(h) + b_g)$$

onde:
- $\theta_0 \approx 155^\circ$ (centro universal comum);
- $\Delta \theta \approx 30^\circ$ (alcance de excursão angular cobrindo Code em $131{,}5^\circ$ e GSM em $164^\circ$);
- $w_g \in \mathbb{R}^{5120}$ é um vetor leve com apenas **$5.120$ parâmetros** adicionais (desprezível frente aos bilhões do modelo).

Dessa forma, o próprio contexto da sequência seleciona suavemente a faixa da estrada, sem necessidade de labels explícitos de tarefa.
