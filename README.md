# Mascaramento de PIIs
## Membros:
- Jean Rodrigues Rocha - RA: 813581
- Leonardo Prado Silva - RA: 813169
- Rafael Gimenez Barbeta - RA: 804318
- Vanderlei Guilherme Andrade de Assis - 802162
- Wilker Silva Ribeiro - 813291

## Dados:
- dataset_original.jsonl - dataset original que foi traduzido do inglês para o português
- dataset_origina+traduzido.csv - dataset original com a adição do texto traduzido, usado para calcular métricas de qualidade de tradução
- dataset_traduzido.csv e dataset_traduzido.pkl - dataset final utilizado no treinamento da redes transformers e validação do modelo baseado em regras

## Scripts Auxiliares:
- preprocessamento.py - funções usadas na tradução e preprocessamento dos dados
- modelo_regras.py - funções usadas para ajudar na construção e validação do modelo baseado em regras
- modelo_transformer.py - funções usadas para validar o modelo baseado em arquitetura transformers

## Notebook:
- PII detection & masking.ipyn - É o arquivo com o notebook usado para organizar o préprocessamento, treinamento do modelo transformers e validação das regras usadas pelo modelo baseado em regras
