import pandas as pd
import json
#from easynmt import EasyNMT
from sklearn.model_selection import train_test_split
import re
import sys
import logging
from faker import Faker
import evaluate
import ast
from transformers import AutoTokenizer

tags_invalidas = [
    "[DATE]", "[TIME]", "[GENDER]",
    "[URL]", "[JOBAREA]", "[JOBTYPE]", "[COMPANYNAME]", "[JOBTITLE]","[COUNTY]","[AGE]",
    "[CURRENCYSYMBOL]", "[AMOUNT]", "[CREDITCARDISSUER]", "[SEX]",
    "[CURRENCY]", "[CURRENCYNAME]", "[CURRENCYCODE]", "[ORDINALDIRECTION]",
    "[MASKEDNUMBER]", "[EYECOLOR]", "[HEIGHT]"
]

tags_validas = [
    "[PREFIX]","[FIRSTNAME]","[LASTNAME]","[PHONEIMEI]","[USERNAME]","[CITY]","[STATE]","[EMAIL]","[STREET]",
    "[SECONDARYADDRESS]","[USERAGENT]","[ACCOUNTNAME]","[ACCOUNTNUMBER]",
    "[CREDITCARDNUMBER]","[CREDITCARDCVV]","[PHONENUMBER]","[IP]","[ETHEREUMADDRESS]",
    "[BITCOINADDRESS]","[MIDDLENAME]","[IBAN]","[VEHICLEVRM]","[DOB]","[PIN]","[PASSWORD]",
    "[LITECOINADDRESS]","[BUILDINGNUMBER]","[ZIPCODE]","[BIC]","[IPV4]","[IPV6]","[MAC]",
    "[NEARBYGPSCOORDINATE]","[VEHICLEVIN]","[SSN]"
]

tags_validas_adaptadas = [
    "[PREFIXO]","[PRIMEIRONOME]","[ULTIMONOME]","[IMEI]","[USUARIO]","[CIDADE]","[ESTADO]","[EMAIL]","[RUA]",
    "[ENDERECOSECUNDARIO]","[USERAGENT]","[NOMECONTA]","[NUMEROCONTA]",
    "[NUMEROCARTAOCREDITO]","[CVVCARTAOCREDITO]","[NUMEROTELEFONE]","[IP]","[ENDERECOETHER]",
    "[ENDERECOBITCOIN]","[NOMEMEIO]","[IBAN]","[VRMVEICULO]","[DATANASCIMENTO]","[PIN]","[SENHA]",
    "[ENDERECOLITECOIN]","[NUMEROPREDIO]","[CEP]","[BIC]","[IPV4]","[IPV6]","[MAC]",
    "[ENDERECOGPSPROXIMO]","[VINVEICULO]","[CPF]"
]

def replace_with_index(text : str, old : str, new : str):
    index = text.find(old)
    if index == -1:
        return text, -1
    replaced_text = text.replace(old, new, 1)
    return replaced_text, index

Faker.seed(246)
class PIIBr:
    def __init__(self):
        self.fake = Faker('pt_BR')
        self.tags = {
            "[PREFIXO]": self.fake.prefix,
            "[PRIMEIRONOME]": lambda : self.fake.first_name().split()[0],
            "[ULTIMONOME]":self.fake.last_name,
            "[IMEI]": None,
            "[USUARIO]": None,
            "[CIDADE]":self.fake.city,
            "[ESTADO]":self.fake.state,
            "[EMAIL]":self.fake.email,
            "[RUA]":self.fake.street_address,
            "[ENDERECOSECUNDARIO]": None,
            "[USERAGENT]":None,
            "[NOMECONTA]":None,
            "[NUMEROCARTAOCREDITO]":None,
            "[CVVCARTAOCREDITO]":None,
            "[NUMEROTELEFONE]":self.fake.phone_number,
            "[IP]":None,
            "[ENDERECOETHER]":None,
            "[ENDERECOBITCOIN]":None,
            "[NOMEMEIO]":lambda : self.fake.first_name().split()[0],
            "[IBAN]":None,
            "[VRMVEICULO]":None,
            "[DATANASCIMENTO]":lambda :self.fake.date().replace("-","/"),
            "[PIN]":None,
            "[SENHA]":None,
            "[ENDERECOLITECOIN]":None,
            "[NUMEROPREDIO]":None,
            "[CEP]":self.fake.postcode,
            "[BIC]":None,
            "[IPV4]":None,
            "[IPV6]":None,
            "[MAC]":None,
            "[ENDERECOGPSPROXIMO]":None,
            "[VINVEICULO]":None,
            "[CPF]":self.fake.cpf,
        }

    def adaptar_pii_br(self,texto :str,privacy_mask_old : list[dict]):
        privacy_mask = []
        all_tags = re.findall(r"\[[A-Z46]+\]",texto)
        for tag in all_tags:
            gen_pii = self.tags.get(tag)
            if not gen_pii:
                tag_en = tags_validas[tags_validas_adaptadas.index(tag)]
                old_tag_dict = list(filter(lambda privacy_mask: privacy_mask["label"] == f"{tag_en[1:-1]}",privacy_mask_old))[0]
                privacy_mask_old.remove(old_tag_dict)
                texto,start_index = replace_with_index(texto,tag,old_tag_dict["value"])
                privacy_mask.append({
                        "value": old_tag_dict["value"],
                        "start": start_index,
                        "end": start_index + len(old_tag_dict["value"]),
                        "label": tag
                    }
                )
            else:  
                pii_data = gen_pii()
                texto,start_index = replace_with_index(texto,tag,pii_data)
                privacy_mask.append({
                        "value": pii_data,
                        "start": start_index,
                        "end": start_index + len(pii_data),
                        "label": tag
                    }
                )
        return texto, privacy_mask

def remover_tags_nao_utilizadas(df : pd.DataFrame):
    df_filtrado = df[~df['target_text'].apply(lambda texto: any(palavra in texto for palavra in tags_invalidas))].reset_index(drop=True)
    return df_filtrado

def substituir_tags_para_traducao(texto:str):
    for i,tag_valida in enumerate(tags_validas):
        texto = texto.replace(tag_valida,f"<XTG{i}>")
    return texto

def resubstituir_tags_pos_traducao(texto:str):
    for i,tag_valida in enumerate(tags_validas_adaptadas):
        texto = texto.replace(f"<XTG{i}>",tag_valida)
    return texto

def gerar_source_text_portugues(linha):
    privacy_mask_old = linha["privacy_mask"]
    texto = linha["target_text_pt"]
    pii_br = PIIBr()
    texto, privacy_mask_adaptado = pii_br.adaptar_pii_br(texto,privacy_mask_old)
    return pd.Series([texto, privacy_mask_adaptado], index=['source_text_pt', 'privacy_mask_pt'])

def gerar_span_labels(linha):
    privacy_mask_pt = linha["privacy_mask_pt"]
    texto = linha["source_text_pt"]
    span_label = []
    idx = 0
    for mask in privacy_mask_pt:
        if (idx - mask["start"]) != 0:
            span_label.append([idx,mask["start"],"O"])
        span_label.append([mask["start"],mask["end"],mask["label"]])
        idx = mask["end"]
    if (idx - len(texto)) != 0:
            span_label.append([idx,len(texto),"O"])
    return span_label

def remover_O_spans(span_labels):
    return [span for span in span_labels if span[2] != 'O']

def span_labels_para_bio_tags(linha, tokenizer):
    text = linha["source_text_pt"]
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offset_mapping = encoding['offset_mapping'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

    bio_tags = ['O'] * len(offset_mapping)
    span_labels = remover_O_spans(linha["span_labels_pt"])

    for start_char, end_char, label in span_labels:
        first_token_found = False
        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0:
                continue

            if start >= start_char and end <= end_char:
                tag = f'B-{label.strip("[]")}' if not first_token_found else f'I-{label.strip("[]")}'
                bio_tags[i] = tag
                first_token_found = True

    return pd.Series([tokens[1:-1], bio_tags[1:-1]], index=['bert_tokens_pt', 'bio_tags_pt'])

def traduzir_para_portugues(df: pd.DataFrame, coluna: str = "target_text",carregar_traduzido=False):
    if not carregar_traduzido:
        logging.info("Iniciando tradução da base...")
        #model = EasyNMT('m2m_100_418M')
        logging.info("Convertendo tags para passar no tradutor...")
        #df["target_text"] = df["target_text"].apply(substituir_tags_para_traducao)
        logging.info("Traduzindo...")
        #df["target_text_pt"] = model.translate(df[coluna].tolist(), source_lang="en", target_lang="pt",show_progress_bar=True)
        logging.info("Restaurando tags originais...")
        df["target_text_pt"] = df["target_text_pt"].apply(resubstituir_tags_pos_traducao)
        logging.info("Convertendo informações mascaras para português e gerando source_text_pt...")
        df[["source_text_pt","privacy_mask_pt"]] = df.apply(gerar_source_text_portugues,axis=1)
        df["span_labels_pt"] = df.apply(gerar_span_labels,axis=1)
        
        tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        df[['bert_tokens_pt', 'bio_tags_pt']] = df.apply(span_labels_para_bio_tags, axis=1, args=(tokenizer,))
        
        df[['source_text_pt', 'target_text_pt', 'privacy_mask_pt', 'span_labels_pt', 'bert_tokens_pt', 'bio_tags_pt']].to_csv("dataframe_traduzido_test.csv")
        df[['source_text_pt', 'target_text_pt', 'privacy_mask_pt', 'span_labels_pt', 'bert_tokens_pt', 'bio_tags_pt']].to_pickle("dataframe_traduzido_test.pkl")
    else:
        logging.warning("Carregando base pré traduzida....")
        df = pd.read_csv("dataset_original+traduzido.csv")
    return df

def selecionar_tuplas_validacao_manual(df : pd.DataFrame,mostrar_apenas_dataset=False):
    logging.info("Selecionando dados para validação manual")
    # Seleciona 120 linhas aleatórias
    amostra = pd.DataFrame(df.sample(n=120, random_state=43)) # random_state para garantir reprodutibilidade
    amostra["target_text"] = amostra["target_text"].apply(resubstituir_tags_pos_traducao)
    if not mostrar_apenas_dataset:
        amostra[:40][["target_text"]].to_csv("validacao_manual1.csv")
        amostra[40:80][["target_text"]].to_csv("validacao_manual2.csv")
        amostra[80:120][["target_text"]].to_csv("validacao_manual3.csv")
    return amostra

def carregar_tuplas_validacao_manual() -> pd.DataFrame:
    logging.info("Carregando tuplas para validação da tradução...")
    validacao_manual = pd.read_csv("validacao_manual_correta.csv")
    return validacao_manual

def avaliar_qualidade_traducao(df : pd.DataFrame,df_validacao : pd.DataFrame):
    bleu = evaluate.load("bleu")
    chrF = evaluate.load("chrf")
    target_text_pt = df["target_text_pt"].to_list()
    target_text_validacao = df_validacao["target_text_pt"].to_list()
    resultados_bleu = bleu.compute(predictions=target_text_pt,references=target_text_validacao)
    resultados_chrF = chrF.compute(predictions=target_text_pt,references=target_text_validacao)
    logging.info(f"Score BLEU: {resultados_bleu["bleu"] * 100 }")
    logging.info(f"Score chrF: {resultados_chrF["score"]}")
    return resultados_bleu, resultados_chrF

def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        stream=sys.stdout
)
    #pd.set_option('display.max_colwidth', None)  
    #pd.set_option('display.max_rows', None) 
    logging.info("Carregando Dataset...")
    df = pd.read_json("english_pii_43k.jsonl", lines=True)
    df_limpo = remover_tags_nao_utilizadas(df)
    df_traduzido = traduzir_para_portugues(df_limpo,carregar_traduzido=True)
    amostra = selecionar_tuplas_validacao_manual(df_traduzido)
    df_validacao = carregar_tuplas_validacao_manual()
    # bons resultados. chrF é mais ideal para português
    resultados_bleu, resultados_chrf = avaliar_qualidade_traducao(amostra,df_validacao)

if __name__ == '__main__':
    main()