# Verificação Formal em Lean 4

## Escopo e Limitações da Formalização

Os arquivos `.lean` neste diretório formalizam a **álgebra de contagem de parâmetros, desigualdades estruturais de orçamentos e relações aritméticas de posto** do projeto MathQwen.

### O que está formalmente verificado no Lean 4:
- [x] Contabilidade de parâmetros do particionamento por cartas locais $\le 9.00\text{B}$ via decisão computable (`decide`).
- [x] Monotonia de postos e subespaços em tipos discretos.
- [x] Restrições algébricas entre dimensões de projeção de Stiefel e Grassmann.

### O que NÃO está formalizado (Limitações Matemáticas):
- [ ] O Teorema Espectral de Eckart-Young-Mirsky em espaços funcionais de Hilbert infinito-dimensionais.
- [ ] A convergência geodésica contínua de variedades Riemannianas sobre dados reais de ativação.
- [ ] A garantia de monotonicidade de perda terminal end-to-end de Transformers causais profundos.

Essas propriedades são tratadas empiricamente nos experimentos em `experiments/` e nos relatórios em `docs/`.
