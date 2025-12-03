# DiabetesGPT RAG Chatbot

---
Um chatbot de Geração Aumentada por Recuperação (RAG) focado em responder perguntas específicas sobre **Diabetes**, utilizando um banco de dados vetorial (`pgvector` no Supabase) e o modelo de linguagem Google Gemini. O projeto utiliza uma interface Streamlit com tema personalizado e inclui um mecanismo de *Fallback* para consultas fora do escopo dos documentos.
---

## 👥 Desenvolvedores

**Dupla responsável pelo projeto:**  
* Larissa Brandim
* Vithor Oliveira 

---
## 🧠 Arquitetura do Projeto

O sistema RAG funciona integrando a base de dados vetorial com a capacidade de geração do LLM:

1. **Indexação:** Documentos na pasta `data/` são convertidos em vetores (embeddings) usando **SentenceTransformer** e armazenados no **Supabase** via `pgvector`.  
2. **Recuperação (Retrieval):** A pergunta do usuário é vetorizada. O Supabase busca os 5 *chunks* mais relevantes através da similaridade de cosseno.  
3. **Geração:** Os *chunks* são injetados no **Google Gemini** como contexto para formular uma resposta precisa, com referências às fontes.

---

## 📂 Estrutura do Projeto

* **app.py:** Contém o código Streamlit (interface de chat), a lógica do ciclo RAG e o Fallback.  
* **src/index_docs.py:** Script para ler documentos, gerar embeddings e popular o Supabase.  
* **data/:** Armazena os documentos de conhecimento (PDFs, TXT) sobre Diabetes.  
* **.env:** Contém as chaves de API do Gemini e Supabase.  
* **.streamlit/config.toml:** Define cores e tema personalizado da interface Streamlit.  
* **requirements.txt:** Lista de bibliotecas Python necessárias (Streamlit, google-genai, etc.).

---

## 🎬 Vídeo do Projeto

[Link do vídeo](https://youtu.be/R0_60B_slws)

---

## 🛠️ Configuração e Instalação

**Requisitos:**  
* Python 3.9+  
* Conta no Supabase com `pgvector` ativado  
* Chave de API do Google Gemini  

**Clonar e configurar ambiente:**  
* Ativar o ambiente: venv\Scripts\activate  
* Instalar dependências: pip install -r requirements.txt  

---

## ⚙️ Uso do Projeto

**Passo 1:** Indexar documentos  
Coloque seus documentos na pasta `data/` e execute:  
python src/index_docs.py  

**Passo 2:** Rodar a aplicação web  
Inicie o Streamlit:  
streamlit run app.py  
A interface abre em http://localhost:8501

---

## ✨ Funcionalidades Avançadas

**Fallback Inteligente:**  
Se a busca RAG não encontrar documentos relevantes, o sistema usa o conhecimento geral do Google Gemini. A interface indica a fonte (`📚 RAG` ou `💡 FALLBACK`).

**Tema Personalizado:**  
Interface com cores rosa forte definida em `.streamlit/config.toml`.

---

## 🧩 Componentes do Projeto

* **Streamlit:** Interface do usuário  
* **Google Gemini:** Modelo de geração de respostas  
* **Supabase + pgvector:** Banco de dados vetorial  
* **SentenceTransformer:** Geração de embeddings  
* **Python Scripts:** `index_docs.py` e `app.py`
