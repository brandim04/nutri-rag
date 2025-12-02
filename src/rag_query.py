# src/rag_query.py (Versão Google Gemini)
import os
import time
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from google import genai # <--- NOVO: Importa o SDK do Google GenAI
from typing import List, Dict, Any

# --- 1. Configuração Inicial e Logs ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- 2. Variáveis de Ambiente e Conexão ---

# Supabase (usado para RAG Query via API/URL)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") 

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # <--- NOVO
LLM_MODEL_NAME = "gemini-2.5-flash" # <--- Modelo rápido e gratuito do Gemini
K_CHUNKS = 5 

# Inicializa o cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    logging.info("Cliente Supabase inicializado.")
except Exception as e:
    logging.error(f"Falha ao inicializar cliente Supabase: {e}")
    supabase = None
    
# Inicializa o cliente Gemini
try:
    # O cliente do Google GenAI usa a variável GEMINI_API_KEY automaticamente
    client_gemini = genai.Client(api_key=GEMINI_API_KEY)
    logging.info("Cliente Google Gemini inicializado.")
except Exception as e:
    logging.error(f"Falha ao inicializar cliente Gemini: {e}")
    client_gemini = None

# Carrega o modelo de embeddings 
try:
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    model_embedding = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    logging.info("Modelo de embeddings carregado para busca.")
except Exception as e:
    logging.error(f"Falha ao carregar modelo de embeddings: {e}")
    model_embedding = None

# --- 3. Funções de Busca RAG ---

def get_relevant_chunks(query_text: str) -> List[str]:
    """
    Gera embedding da query e busca os chunks mais relevantes no Supabase.
    (Esta função permanece a mesma, pois só usa o Supabase)
    """
    if not model_embedding or not supabase:
        logging.error("Modelos ou clientes não inicializados.")
        return []
        
    logging.info(f"Gerando embedding para a query: '{query_text[:30]}...'")
    query_embedding = model_embedding.encode(query_text, normalize_embeddings=True).tolist()
    
    try:
        logging.info(f"Buscando {K_CHUNKS} chunks mais similares...")
        
        response = supabase.rpc(
            'match_documents', 
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.5,
                'match_count': K_CHUNKS
            }
        ).execute()

        relevant_chunks = []
        for match in response.data:
            content_with_source = f"[Fonte: {match['source']}, Score: {match['similarity']:.3f}] {match['content']}"
            relevant_chunks.append(content_with_source)
            
        logging.info(f"Encontrados {len(relevant_chunks)} chunks relevantes.")
        return relevant_chunks
        
    except Exception as e:
        logging.error(f"Erro ao executar busca RAG no Supabase: {e}")
        return []


def generate_rag_answer(query_text: str, relevant_chunks: List[str]) -> str:
    """
    Constrói o prompt final e usa o LLM (Gemini) para gerar a resposta.
    """
    if not client_gemini:
        return "Erro: Cliente Gemini não inicializado. Verifique a chave API no .env."
        
    context = "\n---\n".join(relevant_chunks)
    
    # 1. Definir o System Instruction (Instruções para o LLM)
    system_instruction = (
        "Você é um assistente de Nutrição especializado, chamado NutriRAG. "
        "Sua tarefa é responder a pergunta do usuário APENAS com base nos documentos de contexto fornecidos. "
        "Se o contexto não contiver informações suficientes para responder à pergunta, você deve dizer: 'Desculpe, não encontrei informações relevantes nos documentos fornecidos.' "
        "Mantenha a resposta concisa, profissional e inclua referências à fonte original entre colchetes [Fonte: nome_do_arquivo] no final da frase relevante."
    )
    
    # 2. Definir o User Prompt (Pergunta + Contexto)
    user_prompt = (
        f"Contexto dos Documentos:\n\n{context}\n\n"
        f"---"
        f"Pergunta do Usuário: {query_text}"
    )
    
    logging.info(f"Enviando prompt e contexto para o LLM ({LLM_MODEL_NAME})...")
    
    try:
        # 3. Chamar a API do Google GenAI
        response = client_gemini.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2, 
            )
        )
        
        return response.text
        
    except Exception as e:
        logging.error(f"Erro ao chamar a API do Google Gemini: {e}")
        return "Erro ao gerar resposta com o LLM."

# --- 4. Função Principal (Main) ---

def main():
    """Fluxo principal do RAG Chatbot."""
    
    if not supabase or not client_gemini or not model_embedding:
        logging.error("O programa não pode ser executado devido a falhas na inicialização.")
        return

    print("\n==============================================")
    print("🤖 NutriRAG Chatbot (Gemini) Inicializado!")
    print("==============================================")
    
    while True:
        user_query = input("Pergunte sobre Diabetes (ou 'sair'): ").strip()
        
        if user_query.lower() in ['sair', 'exit', 'quit']:
            print("Chatbot encerrado. Até logo!")
            break
            
        if not user_query:
            continue
            
        start_time = time.time()
        
        # 1. Recuperação (R)
        relevant_chunks = get_relevant_chunks(user_query)
        
        if not relevant_chunks:
            print("\nNutriRAG > Desculpe, não foi possível realizar a busca no banco de dados. Verifique o log.")
            continue
            
        # 2. Geração (G)
        final_answer = generate_rag_answer(user_query, relevant_chunks)
        
        end_time = time.time()
        
        print("\n--- RESPOSTA NUTRI-RAG ---")
        print(final_answer)
        print("--------------------------")
        print(f"Tempo de resposta total: {end_time - start_time:.2f} segundos.")
        print("--------------------------\n")


if __name__ == "__main__":
    main()