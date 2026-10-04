import streamlit as st
import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# COLOQUE AQUI A URL GERADA NO COLAB
COLAB_API_URL = "https://whooping-renovator-keg.ngrok-free.dev/gerar"

def chamar_llm_colab(prompt_formatado):
    # Substitua prompt_formatado.text por prompt_formatado.to_string()
    texto_prompt = prompt_formatado.to_string() 
    
    try:
        resposta = requests.post(COLAB_API_URL, json={"prompt": texto_prompt})
        if resposta.status_code == 200:
            return resposta.json().get("resposta", "Resposta não encontrada.")
        return f"Erro na API do Colab. Status: {resposta.status_code}"
    except Exception as e:
        return f"Falha de conexão com o Colab: {str(e)}"

# Decorator de cache evita que o PDF seja reprocessado toda vez que o usuário envia uma mensagem
@st.cache_resource 
def iniciar_banco_vetorial():
    # 1. Descobre a pasta exata onde este script (main.py) está salvo
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Junta o caminho da pasta com o nome do arquivo PDF
    caminho_pdf = os.path.join(diretorio_atual, "politica_rh.pdf")
    
    # 3. Carrega o PDF usando o caminho completo e absoluto
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    
    # ... (restante do seu código continua igual) ...
    separador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    blocos = separador.split_documents(documentos)
    
    # Embeddings 100% locais e gratuitos (Roda na CPU do servidor de hospedagem)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    return FAISS.from_documents(blocos, embeddings)

banco_vetores = iniciar_banco_vetorial()

prompt_template = ChatPromptTemplate.from_template(
    """Use APENAS as informações do contexto abaixo para responder à pergunta.
    Se não encontrar a resposta no contexto, diga claramente que não sabe. Responda em português do Brasil, de forma clara e objetiva.
    
    Contexto: {context}
    
    Pergunta: {question}
    """
)

# A chain orquestra: Busca FAISS -> Monta Template -> Envia para Colab
chain = (
    {"context": banco_vetores.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()} 
    | prompt_template 
    | RunnableLambda(chamar_llm_colab)
)

# ============================================================
# INTERFACE
# ============================================================
st.title("Assistente RAG — Políticas de RH")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Pergunte sobre as políticas da empresa...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando os documentos e gerando resposta..."):
            answer = chain.invoke(prompt)
            st.write(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})