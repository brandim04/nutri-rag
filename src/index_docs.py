import os
import time
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests 
from supabase import create_client, Client 
from tqdm import tqdm
import logging
import json 
from typing import List, Dict, Any

# --- 1. Configuração Inicial e Logs ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- 2. Variáveis de Ambiente e Conexão (API) ---

# URL e Chave para conexão via REST API
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") 

# Configurações do modelo
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR = "docs"
BATCH_SIZE = 500 

# Endpoint da Tabela e Headers
INSERT_URL = f"{SUPABASE_URL}/rest/v1/documents"
HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}", # Usa a chave de serviço para autenticação
    "Content-Type": "application/json",
    "Prefer": "return=minimal" 
}

# --- 3. Funções de Ajuda ---

def extract_text_from_pdf(filepath: str) -> str:
    """Extrai texto de um arquivo PDF."""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in tqdm(reader.pages, desc=f"Extraindo {os.path.basename(filepath)}"):
            text += page.extract_text() or ""
        logging.info(f"Texto extraído de {filepath}. Tamanho: {len(text)} caracteres.")
        return text
    except Exception as e:
        logging.error(f"Erro ao extrair texto do PDF {filepath}: {e}")
        return ""

def chunk_text(text: str, filename: str) -> List[Dict[str, Any]]:
    """Divide o texto em chunks e prepara metadados."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    
    data = []
    for i, chunk in enumerate(chunks):
        if 'diabetes' in filename.lower():
            doenca = "Diabetes"
        elif 'cancer' in filename.lower() or 'estomago' in filename.lower():
            doenca = "Câncer de Estômago"
        else:
            doenca = "Geral/Não Especificado"
            
        data.append({
            "source": filename,
            "chunk_index": i,
            "content": chunk,
            "metadata": {"doenca": doenca, "tema": "nutrição"}
        })
        
    logging.info(f"Texto dividido em {len(chunks)} chunks.")
    return data

def generate_embeddings(chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gera embeddings para cada chunk de texto."""
    texts = [item['content'] for item in chunks_data]
    
    logging.info(f"Carregando e gerando embeddings para {len(texts)} chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') 
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    for i, embedding in enumerate(embeddings):
        # O REST API aceita o embedding como uma lista Python normal
        chunks_data[i]['embedding'] = embedding.tolist()
        
    logging.info("Geração de embeddings concluída.")
    return chunks_data

def insert_data_to_supabase_api(chunks_data: List[Dict[str, Any]]):
    """Insere os chunks via REST API do Supabase em lotes (ignora problemas de rede/psycopg2)."""
    
    total_records = len(chunks_data)
    logging.info(f"Iniciando inserção de {total_records} registros via REST API em lotes de {BATCH_SIZE}...")
    
    success_count = 0
    
    # Loop para processar os dados em lotes
    for i in range(0, total_records, BATCH_SIZE):
        batch_data = chunks_data[i:i + BATCH_SIZE]
        payload = batch_data
        
        try:
            # Envia o lote via POST (inserção)
            response = requests.post(INSERT_URL, headers=HEADERS, json=payload)
            response.raise_for_status() 
            
            success_count += len(batch_data)
            logging.info(f"Lote {i//BATCH_SIZE + 1} de {len(batch_data)} registros inserido com sucesso (HTTP {response.status_code}).")

        except requests.exceptions.HTTPError as e:
            logging.error(f"Erro HTTP ao inserir Lote {i//BATCH_SIZE + 1}. Código: {e.response.status_code}. Resposta: {e.response.text}")
            logging.error("O erro (401) é devido à chave SUPABASE_SERVICE_KEY incorreta no .env.")
            break
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro de Conexão de Rede durante a inserção. Erro: {e}")
            break
        except Exception as e:
            logging.error(f"Erro inesperado durante a inserção: {e}")
            break

    if success_count == total_records:
        logging.info(f"Indexação COMPLETA! Total de {total_records} registros processados via API.")
    else:
        logging.warning(f"Indexação PARCIAL ou Falha Total. {success_count} de {total_records} inseridos.")


# --- 4. Função Principal (Main) ---

def main():
    """Fluxo principal de indexação."""
    start_time = time.time()
    logging.info("Iniciando processo NutriRAG Indexer (Modo API)...")
    
    # Verifica as chaves da API
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logging.error("Variáveis SUPABASE_URL ou SUPABASE_SERVICE_KEY não encontradas. Verifique seu arquivo .env.")
        return

    # 1. Processar Documentos
    if not os.path.exists(DOCS_DIR):
        logging.warning(f"Diretório '{DOCS_DIR}' não encontrado. Criando...")
        os.makedirs(DOCS_DIR)
        logging.info("🛑 Coloque seus arquivos PDF na pasta 'docs/' e execute o script novamente.")
        return

    total_chunks_data = []
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        logging.warning(f"Nenhum arquivo PDF encontrado no diretório '{DOCS_DIR}'.")
        return

    for filename in pdf_files:
        filepath = os.path.join(DOCS_DIR, filename)
        
        logging.info(f"\n--- Processando Arquivo: {filename} ---")
        
        text = extract_text_from_pdf(filepath)
        if text:
            chunks = chunk_text(text, filename)
            chunks_with_embeddings = generate_embeddings(chunks)
            total_chunks_data.extend(chunks_with_embeddings)
            
    
    # 2. Inserção no Supabase (via API de REST)
    if total_chunks_data:
        logging.info(f"\n--- Iniciando Inserção via REST API de {len(total_chunks_data)} Chunks ---")
        insert_data_to_supabase_api(total_chunks_data)
    else:
        logging.warning("Nenhum dado processado para inserção.")
        
    # 3. Finalização
    end_time = time.time()
    logging.info(f"\nProcesso de indexação concluído em {end_time - start_time:.2f} segundos.")


if __name__ == "__main__":
    main()
