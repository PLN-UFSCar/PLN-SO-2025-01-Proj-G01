# Mascaramento de PIIs
## Membros:
- Jean Rodrigues Rocha - RA: 813581
- Leonardo Prado Silva - RA: 813169
- Rafael Gimenez Barbeta - RA: 804318
- Vanderlei Guilherme Andrade de Assis - 802162
- Wilker Silva Ribeiro - 813291

## Dados:
- dataset_original.jsonl - dataset original que foi traduzido do inglês para o português
- dataset_origina+traduzido.csv - dataset original com a adição do texto traduzido, usado para calcular métricas de qualidade de tradução
- dataset_traduzido.csv e dataset_traduzido.pkl - dataset final utilizado no treinamento da redes transformers e validação do modelo baseado em regras

## Scripts Auxiliares:
- preprocessamento.py - funções usadas na tradução e preprocessamento dos dados
- modelo_regras.py - funções usadas para ajudar na construção e validação do modelo baseado em regras
- modelo_transformer.py - funções usadas para validar o modelo baseado em arquitetura transformers

## Passo a passo
Para executar o projeto localmente siga o seguintes passos.

### Faça o clone do projeto
```
git clone https://github.com/PLN-UFSCar/PLN-SO-2025-01-Proj-G01.git
```

## Instale as dependências necessárias para a execução do projeto
```
pip install transformers datasets evaluate seqeval torch scikit-learn pandas faker spacy sacrebleu
```
## (Observação) 
Para a execução da etapa de tradução será necessário a biblioteca easynmt, que depende de um pacote do fasttext e deve ser construído manualmente por conflitos de instalação.
[Instruções para montagem do pacote](https://github.com/UKPLab/EasyNMT/issues/89#issuecomment-2021129757)


### Abra o Jupyter Notebook
Com as dependências instaladas, inicie o servidor do Jupyter.
```
jupyter notebook
```
Isso abrirá uma aba no seu navegador. 
### Abra o notebook PII detection & masking.ipynb
Clique no arquivo PII detection & masking.ipynb para abri-lo. Dentro do notebook, execute as células para verificar o funcionamento.
