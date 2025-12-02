# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import time
import logging

# --- 1. CONFIGURAÇÃO E DEPENDÊNCIAS DO RAG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Importações de clientes
try:
    from google import genai
    from supabase import create_client, Client
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    st.error(f"Erro de Importação: Instale: 'pip install streamlit google-genai supabase sentence-transformers python-dotenv'")
    st.stop()

# Variáveis (usando chaves do .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
K_CHUNKS = 5 

# --- 2. INICIALIZAÇÃO DE CLIENTES (Cached) ---
@st.cache_resource
def initialize_resources():
    """Inicializa Supabase, Gemini e Modelo de Embeddings."""
    if not (SUPABASE_URL and SUPABASE_ANON_KEY and GEMINI_API_KEY):
        st.error("ERRO CRÍTICO: Chaves de API não encontradas. Verifique o arquivo .env.")
        return None, None, None

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        model_embedding = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        return supabase, client_gemini, model_embedding
    except Exception as e:
        st.error(f"Erro ao inicializar recursos: {e}")
        return None, None, None

supabase, client_gemini, model_embedding = initialize_resources()

# --- 3. FUNÇÕES DE BUSCA E GERAÇÃO COM FALLBACK ---

def get_relevant_chunks(query_text: str) -> list[str]:
    """Gera embedding e busca chunks no Supabase."""
    if not model_embedding or not supabase: return []
        
    try:
        query_embedding = model_embedding.encode(query_text, normalize_embeddings=True).tolist()
        
        response = supabase.rpc(
            'match_documents', 
            {'query_embedding': query_embedding, 'match_threshold': 0.5, 'match_count': K_CHUNKS}
        ).execute()

        relevant_chunks = []
        for match in response.data:
            if match['similarity'] > 0.75: 
                content_with_source = f"[Fonte: {match['source']}, Score: {match['similarity']:.3f}] {match['content']}"
                relevant_chunks.append(content_with_source)
        return relevant_chunks
        
    except Exception as e:
        logging.error(f"Erro ao buscar documentos no Supabase: {e}")
        return []

def generate_rag_answer(query_text: str, relevant_chunks: list[str]) -> tuple[str, str]:
    """Usa Gemini com contexto (RAG) ou conhecimento geral (FALLBACK)."""
    if not client_gemini: 
        return "Erro: Cliente Gemini não inicializado.", "ERROR"

    if relevant_chunks:
        # --- CENÁRIO 1: RAG (Usa os documentos encontrados) ---
        context = "\n---\n".join(relevant_chunks)
        system_instruction = (
            "Você é um assistente de Nutrição especializado, chamado DiabetesGPT. "
            "Responda APENAS com base nos documentos de contexto fornecidos. "
            "Se o contexto não for suficiente, diga: 'Desculpe, não encontrei informações específicas nos documentos de diabetes fornecidos.' "
            "Inclua referências à fonte original no final da frase relevante."
        )
        source = "RAG"
        
    else:
        # --- CENÁRIO 2: FALLBACK (Usa o conhecimento geral do Gemini) ---
        system_instruction = (
            "Você é um assistente de Nutrição e saúde. A busca RAG falhou. Responda à pergunta "
            "diretamente com seu conhecimento geral de forma informativa e profissional."
        )
        source = "FALLBACK"
    
    user_prompt = f"Pergunta do Usuário: {query_text}\n\nContexto (se RAG): {context if relevant_chunks else 'N/A'}"
    
    try:
        response = client_gemini.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4 if source == "FALLBACK" else 0.2,
            )
        )
        return response.text, source
        
    except Exception as e:
        logging.error(f"Erro ao gerar resposta com o LLM: {e}")
        return "Erro ao gerar resposta com o LLM.", "ERROR"

# --- 4. INTERFACE STREAMLIT (LIMPA) ---

st.set_page_config(page_title="Chatbot de Diabetes", layout="wide")

st.title("📚 DiabetesGPT: Chatbot de Busca Aumentada")
st.markdown("##### Baseado nos seus documentos de Diabetes e alimentado por Google Gemini")

if not all([supabase, client_gemini, model_embedding]):
    st.error("A aplicação não pode rodar devido a erros de inicialização. Verifique o console.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Eu sou o DiabetesGPT. Posso responder suas perguntas sobre diabetes."}
    ]

# Exibe histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Processa nova entrada do usuário
if prompt := st.chat_input("Pergunte sobre Diabetes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inicia processamento
    with st.spinner('Buscando e gerando resposta...'):
        chunks = get_relevant_chunks(prompt)
        response_text, source = generate_rag_answer(prompt, chunks)
        
        # Prepara a exibição no chat
        source_icon = "📚" if source == "RAG" else "💡"
        source_label = "RAG (Documentos)" if source == "RAG" else "FALLBACK (Conhecimento Geral)"
        
        full_response_content = (
            f"**Fonte da Resposta: {source_icon} {source_label}**\n\n"
            f"{response_text}"
        )

        # Exibe a resposta do assistente no chat
        with st.chat_message("assistant"):
            st.markdown(full_response_content)

    # Adiciona a resposta completa ao histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})