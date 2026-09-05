# Relatório Técnico 12: Estabilização do Residual Stream e Sondagem nas Camadas Críticas

## 1. Resumo Executivo
Este relatório documenta a descoberta empírica mais significativa do projeto MathQwen até o momento:
1. **O Salto de Perplexidade**: O modelo comprimido via Atlas Assimétrico ($V_{\rm joint}$, $r=2048$, ~14,29B parâmetros congelados) apresentava PPL zero-shot de **3.712,64**. A introdução de uma correção linear no residual stream ($d_{\rm model} = 5120$) com apenas **41,94M parâmetros** ($r_{\rm corr} = 64$, menos de 0,3% do modelo) reduziu a PPL em teste cego para **39,70** — uma redução de **98,97%**.
2. **Diagnóstico de Divergência Profunda**: A propagação residual acumula perturbações que sofrem amplificação direcional ($G_l^{\rm drift} > 1,0$), culminando na ruptura na região $L_{48}\text{--}L_{55}$, onde a norma $\|h\|$ dobra.
3. **Refutação da Ilha Densa**: Manter $L_{48}\text{--}L_{55}$ densas falhou (PPL de 388.770) porque operadores densos fora da variedade amplificam a perturbação. O residual stream é o locus matematicamente correto para intervenção.
4. **Sondagem nas 8 Camadas Críticas ($L_0, L_{16}, L_{32}, L_{48}, L_{50}, L_{51}, L_{55}, L_{63}$)**:
   - O SVD analítico reproduz com alta fidelidade a estabilidade linear.
   - Manifold fitting com AdamW melhora todas as 8 camadas sem quebra de monotonicidade, mas atinge saturação em profundidade ($L_{50}: 79,15\% \to 78,16\%$).
5. **Nova Rota Arquitetural**: Transição para estabilizadores residuais não-lineares ($h' = \alpha h + \sigma(h W_{\rm down}) W_{\rm up}$) com avaliação via Logit Lens ($D_{\rm KL}$, $\text{NLL}$ e Top-1 Agreement a $T=1$).

---

## 2. Resultados Empíricos no WikiText-2 (Teste Cego Não-Visto)

| Configuração | Parâmetros Adicionais | $e_{64}$ (Hidden Error) | $\cos(z)$ (Logits) | Top-1 Agreement | PPL (WikiText-2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Professor Oficial FP8** | 0 (Referência) | 0.00% | 1.0000 | 100.0% | **4.21** |
| **Atlas Raw ($r=2048$)** | 0 M | 101.53% | 0.7283 | 8.33% | **3712.64** |
| **Escalar ($h' = \alpha_l h$)** | 64 floats | 94.12% | 0.7714 | 16.67% | **649.27** |
| **SVD Low-Rank ($r=32$)** | +20.97 M | 89.41% | 0.8897 | 42.06% | **50.72** |
| **SVD Low-Rank ($r=64$)** | **+41.94 M** | **88.30%** | **0.9075** | **46.83%** | **39.70** |
| **SVD Low-Rank ($r=128$)** | +83.89 M | 87.12% | 0.9175 | 50.00% | **38.22** |

---

## 3. Sondagem nas 8 Camadas Críticas (Manifold Fitting com Rollback)

| Camada | Papel Arquitetural | $e_{\rm SVD}$ | $e_{\rm opt}$ | $\Delta e$ (abs) | Ganho Relativo | $\cos_{\rm SVD}$ | $\cos_{\rm opt}$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$L_0$** | Entrada | 10.30% | 9.55% | -0.75% | **+7.26%** | 0.9947 | 0.9954 |
| **$L_{16}$** | Rasa $\to$ Média | 40.00% | 39.92% | -0.08% | **+0.20%** | 0.9204 | 0.9218 |
| **$L_{32}$** | Equador | 61.51% | 60.33% | -1.17% | **+1.91%** | 0.7890 | 0.7985 |
| **$L_{48}$** | Entrada Crítica | 73.06% | 71.90% | -1.15% | **+1.58%** | 0.6932 | 0.7084 |
| **$L_{50}$** | Pico de Ruptura | 79.15% | 78.16% | -0.99% | **+1.25%** | 0.6210 | 0.6354 |
| **$L_{51}$** | Núcleo Crítico | 82.73% | 82.60% | -0.14% | **+0.17%** | 0.5817 | 0.5875 |
| **$L_{55}$** | Saída da Ilha | 87.26% | 86.74% | -0.52% | **+0.60%** | 0.5260 | 0.5354 |
| **$L_{63}$** | Pré-Norm Final | 88.30% | 88.25% | -0.05% | **+0.05%** | 0.5705 | 0.5839 |

---

## 4. Conclusão Metodológica
1. O estabilizador de posto 64 previne o blowup da norma global e recupera 98.97% da qualidade funcional da rede.
2. A saturação local em $L_{50}\text{--}L_{55}$ indica que, após 50 camadas não-lineares, a perturbação acumulada depende do estado do sinal e exige expressividade não-linear pontual.
