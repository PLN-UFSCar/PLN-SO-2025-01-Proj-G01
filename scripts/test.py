import pandas as pd
import pickle
import os
import sys
from typing import Union, Dict, Any, List, Tuple
from faker import Faker
import re
import numpy as np

# --- Configuration and Helper Functions (Largely Unchanged) ---

fake = Faker('pt_BR')

regex_tags = {
    "[IPV4]": r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\b",
    "[CPF]": r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
    "[BIC]": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
    "[IMEI]": r"\b\d{15}\b",
    "[EMAIL]": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
    "[CEP]": r"\b\d{5}-\d{3}\b",
    "[IPV6]": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    "[MAC]": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
    "[VINVEICULO]": r"\b[A-HJ-NPR-Z0-9]{17}\b",
    "[ENDERECOLITECOIN]": r"\b[L3M][a-km-zA-HJ-NP-Z1-9]{26,33}\b",
    "[ENDERECOETHER]": r"\b0x[a-fA-F0-9]{40}\b",
    "[ENDERECOBITCOIN]": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
    "[ENDERECOSECUNDARIO]": r"(?i)\b(?:apt\.?|apto\.?|bloco|andar|unidade)\s*\d+\b",
    "[ENDERECOGPSPROXIMO]": r"\[\s*-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\s*\]",
    "[NUMEROCARTAOCREDITO]": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
    "[NUMEROTELEFONE]": r"\b\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b",
    "[IBAN]": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    "[VRMVEICULO]": r"\b[A-Z]{3}-?\d{4}\b",
    "[DATANASCIMENTO]": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    "[RUA]": r"\b(?:Rua|Av\.?|Avenida|Travessa|Alameda|Rodovia)\s+[^,\n\[\]]+",
    "[USERAGENT]": r"Mozilla/\d+\.\d+\s*\([^)]+\)\s*[^\n\[\]]+",
    "[PREFIXO]": r"(?i)\b(?:sr\.?|sra\.?|dr\.?|dra\.?|srta\.?)\b",
}

context_patterns = {
    "[NUMEROCONTA]": r"\b\d{6,12}\b",
    "[PIN]": r"\b\d{4,6}\b",
    "[CVVCARTAOCREDITO]": r"\b\d{3,4}\b",
    "[NUMEROPREDIO]": r"(?<=,\s*)\b\d{1,5}\b"
}

ESTADOS_BRASILEIROS = {
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO',
    'Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará', 'Distrito Federal', 'Espírito Santo', 'Goiás', 'Maranhão', 'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais', 'Pará', 'Paraíba', 'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro', 'Rio Grande do Norte', 'Rio Grande do Sul', 'Rondônia', 'Roraima', 'Santa Catarina', 'São Paulo', 'Sergipe', 'Tocantins'
}

def carregar_senhas_rockyou(arquivo_path="rockyou.txt", max_senhas=10000):
    senhas = set()
    try:
        with open(arquivo_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, linha in enumerate(f):
                if i >= max_senhas: break
                senha = linha.strip()
                if len(senha) >= 4 and not senha.isdigit():
                    senhas.add(senha)
    except FileNotFoundError:
        print(f"Warning: '{arquivo_path}' not found. Using default common passwords.")
        senhas = {"password", "123456789", "qwerty", "admin", "12345"}
    return senhas

def is_valid_ipv4(ip):
    try:
        parts = ip.split('.')
        return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
    except: return False

def is_valid_date(date_str):
    try:
        parts = date_str.split('/')
        if len(parts) != 3: return False
        day, month, year = map(int, parts)
        return 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2030
    except: return False

def is_likely_phone(phone):
    digits = re.sub(r'[^\d]', '', phone)
    return len(digits) in [10, 11]

# --- REFACTORED SPAN EXTRACTION FUNCTIONS ---

def extrair_spans_regex(texto: str, tag: str, pattern: str) -> List[Tuple[int, int, str]]:
    """Applies regex with validation and returns a list of spans (start, end, tag)."""
    spans = []
    if not isinstance(texto, str): return spans
    
    for match in re.finditer(pattern, texto):
        matched_text = match.group()
        if tag == "[IPV4]" and not is_valid_ipv4(matched_text): continue
        if tag == "[DATANASCIMENTO]" and not is_valid_date(matched_text): continue
        if tag == "[NUMEROTELEFONE]" and not is_likely_phone(matched_text): continue
        if tag == "[CEP]" and not re.match(r'^\d{5}-\d{3}$', matched_text): continue
        
        start, end = match.span()
        spans.append((start, end, tag.strip("[]")))
    return spans

def extrair_spans_contexto(texto: str) -> List[Tuple[int, int, str]]:
    """Detects patterns in specific contexts and returns their spans."""
    spans = []
    if not isinstance(texto, str): return spans
    
    context_map = {
        "conta": (['conta', 'account', 'número da conta'], context_patterns["[NUMEROCONTA]"], "[NUMEROCONTA]"),
        "pin": (['pin:', 'pin ', 'código'], context_patterns["[PIN]"], "[PIN]"),
        "cvv": (['cvv', 'código de segurança'], context_patterns["[CVVCARTAOCREDITO]"], "[CVVCARTAOCREDITO]"),
        "predio": (['endereço', 'rua', 'av.', 'avenida'], context_patterns["[NUMEROPREDIO]"], "[NUMEROPREDIO]")
    }
    
    current_offset = 0
    for linha in texto.split('\n'):
        linha_lower = linha.lower()
        for _, (keywords, pattern, tag) in context_map.items():
            if any(word in linha_lower for word in keywords):
                for match in re.finditer(pattern, linha):
                    start, end = match.span()
                    spans.append((start + current_offset, end + current_offset, tag.strip("[]")))
        current_offset += len(linha) + 1
    return spans

def extrair_spans_from_list(texto: str, label: str, word_list: set, context_keywords: List[str]) -> List[Tuple[int, int, str]]:
    """Generic function to find words from a list in a specific context."""
    spans = []
    if not isinstance(texto, str): return spans

    current_offset = 0
    sorted_words = sorted(list(word_list), key=len, reverse=True)
    for linha in texto.split('\n'):
        linha_lower = linha.lower()
        if any(ctx in linha_lower for ctx in context_keywords):
            for word in sorted_words:
                try:
                    pattern = r'\b' + re.escape(word) + r'\b'
                    for match in re.finditer(pattern, linha, flags=re.IGNORECASE):
                        start, end = match.span()
                        spans.append((start + current_offset, end + current_offset, label))
                except re.error:
                    continue # Skip words that might create invalid regex
        current_offset += len(linha) + 1
    return spans

def extrair_spans_usuarios(texto: str) -> List[Tuple[int, int, str]]:
    """Detects usernames in appropriate contexts."""
    spans = []
    if not isinstance(texto, str): return spans

    current_offset = 0
    for linha in texto.split('\n'):
        linha_lower = linha.lower()
        if any(ctx in linha_lower for ctx in ['user:', 'username:', 'usuário:', 'usuario:']):
            pattern = r'\b[a-zA-Z][a-zA-Z0-9_\.]{2,19}\b'
            exclude_list = {'user', 'username', 'admin', 'system', 'successful', 'failed', 'attempt'}
            for match in re.finditer(pattern, linha):
                if match.group().lower() not in exclude_list:
                    start, end = match.span()
                    spans.append((start + current_offset, end + current_offset, "USUARIO"))
        current_offset += len(linha) + 1
    return spans

def remover_spans_sobrepostos(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Removes spans that are contained within other, longer spans."""
    if not spans: return []

    spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    
    result = []
    if spans:
        last_span = spans[0]
        for i in range(1, len(spans)):
            current_span = spans[i]
            if current_span[1] <= last_span[1]:
                continue
            else:
                result.append(last_span)
                last_span = current_span
        result.append(last_span)
    return result

def extrair_spans_do_texto(texto: str, senhas_rockyou: set) -> List[Tuple[int, int, str]]:
    """Applies all detection rules and returns a clean list of spans."""
    if not isinstance(texto, str):
        texto = str(texto)

    all_spans = []

    # 1. Context-specific rules first
    all_spans.extend(extrair_spans_contexto(texto))

    # 2. Rules based on lists and specific contexts
    all_spans.extend(extrair_spans_from_list(texto, "ESTADO", ESTADOS_BRASILEIROS, ['estado:', 'state:']))
    all_spans.extend(extrair_spans_from_list(texto, "SENHA", senhas_rockyou, ['senha', 'password', 'pass', 'pwd']))
    all_spans.extend(extrair_spans_from_list(texto, "CIDADE", {'São Paulo', 'Rio de Janeiro', 'Brasília', 'Salvador', 'Fortaleza'}, ['cidade:']))
    all_spans.extend(extrair_spans_usuarios(texto))
    # You can add the 'nomes' function here if needed, following the same pattern

    # 3. General Regex rules
    ordem_aplicacao = [
        "[EMAIL]", "[CPF]", "[NUMEROTELEFONE]", "[CEP]", "[DATANASCIMENTO]",
        "[IPV4]", "[IPV6]", "[MAC]", "[USERAGENT]", "[RUA]", "[BIC]", "[IBAN]",
        "[ENDERECOBITCOIN]", "[ENDERECOLITECOIN]", "[ENDERECOETHER]",
        "[VINVEICULO]", "[VRMVEICULO]", "[ENDERECOSECUNDARIO]", "[PREFIXO]",
        "[NUMEROCARTAOCREDITO]", "[ENDERECOGPSPROXIMO]", "[IMEI]"
    ]
    for tag in ordem_aplicacao:
        if tag in regex_tags:
            all_spans.extend(extrair_spans_regex(texto, tag, regex_tags[tag]))
    
    # 4. Clean up overlaps
    return remover_spans_sobrepostos(all_spans)


# --- UPDATED DATAFRAME PROCESSING ---

def processar_dataframe_para_spans(df: pd.DataFrame, senhas_rockyou: set) -> pd.DataFrame:
    """Processes a DataFrame to add a new column with PII spans."""
    df_annotated = df.copy()
    if 'source_text_pt' in df_annotated.columns:
        print("Extracting spans from column: source_text_pt")
        df_annotated['spans_pt'] = df_annotated['source_text_pt'].apply(
            lambda x: extrair_spans_do_texto(x, senhas_rockyou) if pd.notna(x) else []
        )
    else:
        print("Column 'source_text_pt' not found in DataFrame.")
    return df_annotated


# --- MAIN EXECUTION BLOCK ---

def carregar_dataset(caminho_arquivo: str) -> Any:
    """Loads a dataset from a PKL file."""
    try:
        with open(caminho_arquivo, 'rb') as f:
            data = pickle.load(f)
        print(f"Dataset loaded successfully: {caminho_arquivo}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def salvar_dataset(data: Any, caminho_arquivo: str) -> None:
    """Saves a dataset to a PKL file."""
    try:
        with open(caminho_arquivo, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved successfully: {caminho_arquivo}")
    except Exception as e:
        print(f"Error saving dataset: {e}")
        raise

def main():
    """Main function to process a PKL dataset and extract spans."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.pkl> <output_annotated_file.pkl>")
        sys.exit(1)
        
    arquivo_entrada = sys.argv[1]
    arquivo_saida = sys.argv[2]
    
    if not os.path.exists(arquivo_entrada):
        print(f"Error: Input file '{arquivo_entrada}' not found!")
        sys.exit(1)
    
    print(f"Loading rockyou passwords...")
    senhas_rockyou = carregar_senhas_rockyou()
    
    print(f"Starting processing of {arquivo_entrada}")
    print("="*50)
    
    data_original = carregar_dataset(arquivo_entrada)
    
    print("Extracting PII spans...")
    data_annotated = processar_dataframe_para_spans(data_original, senhas_rockyou)
    
    salvar_dataset(data_annotated, arquivo_saida)
    
    print("="*50)
    print("SPAN EXTRACTION REPORT")
    print("="*50)
    total_spans = data_annotated['spans_pt'].apply(len).sum()
    print(f"Total rows processed: {len(data_annotated)}")
    print(f"Total PII spans found: {total_spans}")
    
    print("\nProcessing complete!")
    print(f"Original file: {arquivo_entrada}")
    print(f"Annotated file saved to: {arquivo_saida}")

if __name__ == '__main__':
    main()