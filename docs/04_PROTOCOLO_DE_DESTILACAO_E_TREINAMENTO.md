# 04: Protocolo de Destilação e Treinamento: Arquitetura $\mathcal{G}$-Qwen 9B

**Data:** 04 de Setembro de 2026  
**Autores:** Antigravity AI & Nyx  
**Modelo Alvo:** $\mathcal{G}$-Qwen 9B (Compressão de Qwen 3.8 27B-FP8 via Atlas Bipartido de Stiefel)  
**Ambiente de Execução:** Local (NVIDIA RTX 3060 12GB) e Nuvem (1x A100 80GB SXM)

---

## 1. Formulação da Função de Perda de Atlas

A transferência de conhecimento do **Qwen 3.8 27B** para a arquitetura **$\mathcal{G}$-Qwen 9B** explora a geometria das 16 cartas locais para garantir fidelidade de subespaço e convergência acelerada:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KD}} + \lambda_1 \mathcal{L}_{\text{Fidelity}} + \lambda_2 \mathcal{L}_{\text{SheafTransition}} + \lambda_3 \mathcal{L}_{\text{StiefelReg}}$$

### 1.1 Perda de Destilação de Logits ($\mathcal{L}_{\text{KD}}$)
Divergência de Kullback-Leibler inversa suavizada entre os logits do professor (27B) e do estudante (9B):
$$\mathcal{L}_{\text{KD}} = T^2 \cdot \mathcal{D}_{\mathrm{KL}}\left( \mathrm{softmax}\left(\frac{z_{27\text{B}}}{T}\right) \;\Bigg\|\; \mathrm{softmax}\left(\frac{z_{9\text{B}}}{T}\right) \right)$$
com temperatura de suavização $T = 1.5$.

### 1.2 Perda de Fidelidade de Ativação Camada a Camada ($\mathcal{L}_{\text{Fidelity}}$)
Como todas as 64 camadas existem ativamente em uma passada direta $1:1$ (sem loops artificiais), o alinhamento de representação intermediária é estritamente pontual:
$$\mathcal{L}_{\text{Fidelity}} = \sum_{l=1}^{64} \left\| h_{9\text{B}}^{(l)} - h_{27\text{B}}^{(l)} \right\|_{\Sigma_l^{-1}}^2$$
onde $\Sigma_l$ é a matriz diagonal de variância empírica dos canais na camada $l$.

### 1.3 Perda de Continuidade de Feixe na Transição entre Cartas ($\mathcal{L}_{\text{SheafTransition}}$)
Garante que na passagem de uma carta $g$ para a carta seguinte $g+1$ (isto é, na fronteira entre a camada $4g$ e a camada $4g+1$), o fluxo de ativação preserve a isotopia de representação:
$$\mathcal{L}_{\text{SheafTransition}} = \sum_{g=1}^{15} \left\| h_{9\text{B}}^{(4g+1)} - \Phi_{g \to g+1}(h_{9\text{B}}^{(4g)}) \right\|_2^2$$
onde $\Phi_{g \to g+1}$ é o mapa de transição ortogonal em $\mathrm{SO}(d)$.

### 1.4 Regularizador de Ortogonalidade de Stiefel ($\mathcal{L}_{\text{StiefelReg}}$)
Mantém as bases $\mathbf{U}_g$ e $\mathbf{V}_g$ rigorosamente sobre a variedade de Stiefel $\mathrm{St}(d, r)$:
$$\mathcal{L}_{\text{StiefelReg}} = \sum_{g=1}^{16} \left( \|\mathbf{U}_g^T \mathbf{U}_g - I_r\|_F^2 + \|\mathbf{V}_g^T \mathbf{V}_g - I_r\|_F^2 \right)$$

---

## 2. Cronograma de Treinamento em Três Fases

### Fase 1: Inicialização Analítica sem Treinamento — [CONCLUÍDA]
* Execução do pipeline de Gramianos e SVD em `tools/extract_g_qwen_phase1.py`.
* **Resultado:** O $\mathcal{G}$-Qwen 9B foi inicializado com **mais de $96.8\%$ de fidelidade espectral** sem nenhum passo de retropropagação.
* **Artefatos:** Checkpoints salvos em [`models/g_qwen_9b_phase1/`](file:///C:/Users/Nyx/Desktop/MathQwen/models/g_qwen_9b_phase1).

### Fase 2: Calibração Analítica das Coordenadas via Hessiana (AWQ-Style) — [CONCLUÍDA]
* Execução da solução analítica de mínimos quadrados em `tools/calibrate_phase2_hessian.py`.
* **Formula Exata:**
  $$S_l^* = (U_g^T E_l H V_g) (V_g^T H V_g + \lambda I)^{-1}$$
* **Resultado:** Calibração completa das 64 camadas e 16 cartas em **268.53 segundos** na RTX 3060, com pico de VRAM de **1.63 GB**.
* **Validação:** Geração validada com o clone de Minecraft em Three.js (`generated_minecraft_by_g_qwen9b.html`).
* **Artefatos:** Checkpoints salvos em [`models/g_qwen_9b_phase2/`](file:///C:/Users/Nyx/Desktop/MathQwen/models/g_qwen_9b_phase2).

### Fase 3: Destilação de Ponta a Ponta com Otimização Riemanniana — [PLANEJADA]
* **Objetivo:** Refinamento dos últimos décimos de perplexidade via destilação com o professor 27B denso em cluster de nuvem (1x A100 80GB por ~22h).
* **Parâmetros Treináveis:** Coordenadas $S_l$ e mapas de transição `SheafChartNorm`.
* **Gradiente Riemanniano na Variedade de Stiefel:**
  $$\mathrm{grad}_{\mathrm{St}} \mathcal{L} = \nabla_{\mathbf{U}} \mathcal{L} - \mathbf{U} \cdot \mathrm{sym}(\mathbf{U}^T \nabla_{\mathbf{U}} \mathcal{L})$$
  garantindo que as bases de Stiefel evoluam suavemente ao longo de geodésicas sem violar a ortonormalidade.

---

## 3. Vantagens Computacionais em Relação ao Loop Triplo

1. **Velocidade de Treinamento e Inferência ($3\times$ mais rápido):**
   Como não há repetição de camadas nem desdobramento de loops recorrentes no tempo, o custo de backward pass requer $3\times$ menos memória de ativação e tempo de computação.
2. **Compatibilidade com FlashAttention-3 e vLLM:**
   A arquitetura se comporta como um Transformer padrão de 64 camadas com projeções lineares fatoradas, sendo $100\%$ compatível com paginação de KV-cache (PagedAttention) e paralelismo de tensor (Tensor Parallelism TP=1, 2, 4, 8).
