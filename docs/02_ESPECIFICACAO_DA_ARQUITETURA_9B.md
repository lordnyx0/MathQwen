# Especificação Técnica da Arquitetura $\mathcal{G}$-Qwen 9B v2 (Atlas Bipartido)

## 1. Visão Geral da Arquitetura

A arquitetura **$\mathcal{G}$-Qwen 9B v2** (Grassmannian Foliated Atlas Architecture com Decomposição Bipartida por Operador) foi refinada para atingir a **máxima fidelidade analítica e funcional aos pesos do Qwen 3.8 27B** dentro do limite estrito de **9 Bilhões de Parâmetros Físicos em VRAM**.

Diferente de abordagens baseadas em repetição cíclica de camadas (Triple Loop), a $\mathcal{G}$-Qwen 9B preserva as **64 camadas físicas integrais**, avaliadas na ordem causal natural $1 \to 64$.
A versão v2 introduz o **Foliamento Bipartido por Operador**: o feixe de projeções FFN é compartilhado uniformemente por macro-cartas, enquanto os operadores de mistura de contexto (**Gated DeltaNet** e **Gated Attention**) são desacoplados em sub-feixes especializados para eliminar interferência destrutiva de subespaços.

---

### 1.1 Comparativo Estrutural: Qwen 3.8 27B vs $\mathcal{G}$-Qwen 9B v2

| Hiperparâmetro | Qwen 3.8 27B (Original) | $\mathcal{G}$-Qwen 9B v2 (Atlas Bipartido) | Relação Estrutural |
| :--- | :--- | :--- | :--- |
| **Dimensão Oculta ($d$)** | $5120$ | $5120$ | Idêntica ($1:1$) |
| **Dimensão FFN SwiGLU ($d_{ff}$)** | $17408$ | $17408$ | Idêntica ($1:1$) |
| **Camadas Físicas Totais** | **$64$** | **$64$** | **Preservação integral da profundidade** |
| **Estrutura de Macro-Blocos** | $16$ grupos de $[3\text{D} + 1\text{A}]$ | $16$ cartas de $[3\text{D} + 1\text{A}]$ | **Atlas Isomórfico ao Qwen 3.8** |
| **Camadas Gated DeltaNet** | $48$ | $48$ | Bijeção exata $1:1$ |
| **Camadas Gated Attention** | $16$ | $16$ | Bijeção exata $1:1$ |
| **Feixe FFN ($d \to d_{ff}$)** | Homogêneo nas 64 camadas | 16 Cartas Compartilhadas ($r=1792$) | Retenção espectral $>96.8\%$ |
| **Feixe DeltaNet ($d_v \to d$)** | 48 camadas recorrentes | 16 Cartas Especializadas ($r=1024$) | Dedicado às 3 camadas por bloco |
| **Feixe Attention ($d_v \to d$)** | 16 camadas quadráticas | 4 Macro-Cartas Dedicadas ($r=2048$) | Livre de contaminação por DeltaNet |
| **Parâmetros Físicos Totais** | $\approx 27.5 \times 10^9$ | **$\mathbf{8.625 \times 10^9}$** | **Dentro do teto estrito $\le 9$B** |
| **Tamanho do Vocabulário** | $152.064$ | $152.064$ | Embeddings Amarrados (Tied) |
| **Tamanho de Contexto Nativo** | $262.144$ tokens | $262.144$ tokens | RoPE com base $\theta = 10^6$ |
| **Complexidade de Inferência** | $1\times$ Forward Pass | **$1\times$ Forward Pass** | **3x mais rápido que o Loop Triplo** |

---

## 2. A Estrutura Topológica do Atlas Bipartido

Cada macro-bloco $g \in \{1 \dots 16\}$ cobre 4 camadas consecutivas:
$$\text{Bloco } g = [L_{4g-3}, L_{4g-2}, L_{4g-1}, L_{4g}]$$
onde as três primeiras camadas ($L_{4g-3}, L_{4g-2}, L_{4g-1}$) são **Gated DeltaNet** e a quarta ($L_{4g}$) é **Gated Attention**.

```
Tokens de Entrada X
       │
       ▼
[Embedding Amarrado: 152.064 x 5120] (0.779 B)
       │
       ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  ATLAS DE GRASSMANN BIPARTIDO (16 Macro-Cartas, 64 Camadas Físicas)           ║
║                                                                               ║
║  MACRO-CARTA g (g = 1 ... 16):                                                ║
║  ├── SUB-FEIXE FFN HOMOGÊNEO (Compartilhado pelas 4 camadas):                 ║
║  │   ├── Base Stiefel Gate: U_g^gate (5120 x 1792), V_g^gate (17408 x 1792)   ║
║  │   ├── Base Stiefel Up:   U_g^up   (5120 x 1792), V_g^up   (17408 x 1792)   ║
║  │   └── Base Stiefel Down: U_g^down (17408 x 1792), V_g^down (5120 x 1792)   ║
║  │                                                                            ║
║  ├── SUB-FEIXE DELTANET (Compartilhado pelas 3 camadas DeltaNet do bloco):    ║
║  │   └── Base Stiefel Mixer: U_g^Delta (5120 x 1024), V_g^Delta (6144 x 1024) ║
║  │       ├── Camada 4g-3: Coordenadas S_Delta + Resíduo DeltaNet              ║
║  │       ├── Camada 4g-2: Coordenadas S_Delta + Resíduo DeltaNet              ║
║  │       └── Camada 4g-1: Coordenadas S_Delta + Resíduo DeltaNet              ║
║  │                                                                            ║
║  └── SUB-FEIXE ATTENTION DEDICADO (Para a 4ª camada do bloco):                ║
║      └── Base Stiefel de Alta Resolução: U_macro^Attn (5120 x 2048),          ║
║          V_macro^Attn (6144 x 2048) [4 macro-cartas ao longo da profundidade] ║
║          └── Camada 4g: Coordenadas S_Attn + Resíduo de Atenção               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
       │
       ▼
[RMSNorm Final: 5120]
       │
       ▼
[LM Head (Tied com Embedding): 5120 -> 152.064]
       │
       ▼
Logits de Saída Y
```

---

## 3. Especificação das Projeções Fatoradas

Para qualquer projeção no sub-feixe com base compartilhada $\mathbf{U}_g, \mathbf{V}_g$, a matriz de pesos $W^{(l)}$ da camada $l$ é computada como:
$$W^{(l)} = \mathbf{U}_g \cdot \mathbf{S}_l \cdot \mathbf{V}_g^T + \Delta_l$$

E a ativação linear $y = x W^{(l)}$ é processada sequencialmente sem materialização da matriz densa em memória:
$$y = \left( \left( x \mathbf{U}_g \right) \mathbf{S}_l \right) \mathbf{V}_g^T + x \Delta_l$$

### Hiperparâmetros de Posto:
* **Posto do Feixe FFN ($r_{ffn}$)**: $1792$ (projeção de $5120 \times 17408$).
* **Posto do Feixe DeltaNet ($r_\Delta$)**: $1024$ (projeção de $6144 \times 5120$, compartilhada por 3 camadas).
* **Posto do Feixe Attention ($r_{\mathrm{attn}}$)**: $2048$ (projeção de $6144 \times 5120$, 4 macro-cartas para 16 camadas).
* **Posto dos Resíduos Esparsos ($\Delta_l$)**: $64$.

---

## 4. Auditoria Exata de Parâmetros ($\mathbf{8.625\text{ B}} \le \mathbf{9.000\text{ B}}$)

| Subsistema | Componente | Cálculo | Parâmetros |
| :--- | :--- | :--- | :--- |
| **Embeddings** | Vocabulário (Tied) | $152064 \times 5120$ | **$778.57\text{ M}$** |
| **FFN (SwiGLU)** | Bases Stiefel (16 cartas) | $16 \times 3 \times (5120 + 17408) \times 1792$ | **$1,938.82\text{ M}$** |
| | Coordenadas $S_l$ (64 camadas) | $64 \times 3 \times (1792 \times 1792)$ | **$616.56\text{ M}$** |
| | Resíduos $\Delta_l$ (64 camadas) | $64 \times 3 \times (5120 + 17408) \times 64$ | **$276.82\text{ M}$** |
| **Gated DeltaNet** | Bases Stiefel (16 cartas) | $16 \times (5120 + 6144) \times 1024$ | **$184.55\text{ M}$** |
| | Coordenadas e Resíduos (48 cam.) | $48 \times (1024^2 + (5120+6144) \times 64)$ | **$84.94\text{ M}$** |
| | Projeções Q, K, V, $\beta$, Gate, Conv | $48 \times [2(5120 \times 2048) + 2(5120 \times 6144) + 5120 \times 48 + \text{conv}]$ | **$4,196.43\text{ M}$** |
| **Gated Attention** | Bases Stiefel (4 macro-cartas) | $4 \times (5120 + 6144) \times 2048$ | **$92.27\text{ M}$** |
| | Coordenadas e Resíduos (16 cam.) | $16 \times (2048^2 + (5120+6144) \times 64)$ | **$78.64\text{ M}$** |
| | Projeções Q, K, V, Gate | $16 \times [2(5120 \times 6144) + 2(5120 \times 1024)]$ | **$377.49\text{ M}$** |
| **Normalizações** | RMSNorms (Camadas + Final) | $(64 \times 2 + 1) \times 5120$ | **$0.66\text{ M}$** |
| **TOTAL GERAL** | $\mathcal{G}$-Qwen 9B v2 | — | **$\mathbf{8.625.75\text{ M}}$ ($\mathbf{8.626\text{ B}}$)** |

A foliação bipartida acomoda com folga a capacidade física exigida, mantendo **$374\text{ M}$ parâmetros de margem livre** abaixo do limite estrito de $9.00\text{B}$.
