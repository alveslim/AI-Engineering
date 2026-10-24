# AI Engineering

Repositório de estudos e projetos práticos em **Engenharia de Inteligência Artificial**, com foco em aplicações de IA generativa, sistemas de recuperação de informação (RAG), processamento de linguagem natural e análise de dados com Python.

A ideia é reunir experimentos, notebooks e aplicações que demonstrem como conceitos de IA podem ser aplicados a problemas reais.

## 📚 Conteúdo

### 🤖 Assistentes RAG para políticas de RH

Projetos de assistentes conversacionais que consultam um documento PDF de políticas internas para responder perguntas usando o conteúdo recuperado como contexto.

- **[Assistente RAG — Colab](./Assistente%20RAG%20-%20Colab/)** — interface em Streamlit, processamento de PDF, embeddings multilíngues locais com Sentence Transformers, índice vetorial FAISS e integração com uma API executada no Google Colab.
- **[Assistente RAG — Políticas Internas](./Assistente%20RAG%20-%20Politicas%20Internas/)** — versão com Streamlit, LangChain, FAISS e modelos da OpenAI para embeddings e geração de respostas.

**Fluxo geral:** documento PDF → extração do texto → divisão em trechos → vetorização e indexação → recuperação de contexto relevante → geração da resposta.

Os prompts orientam o assistente a responder em português e a informar quando não encontrar a resposta no contexto, reduzindo respostas sem suporte documental. Isso não substitui validação humana nem garante ausência de erros.

### 🌐 Projetos web com IA

Aplicações web voltadas à saúde, publicadas na Vercel, que incorporam recursos de inteligência artificial.

- **[Segurança do Paciente](https://seguranca-do-paciente.vercel.app/)** — site com informações, boas práticas e os seis protocolos de segurança do paciente, voltado a profissionais, instituições de saúde e pacientes. Conta com o **Assistente de Saúde**, um chatbot com IA que responde dúvidas sobre protocolos e cuidados.
- **[Além do Novembro Azul](https://alem-do-novembro-azul.vercel.app/)** — site sobre a fisioterapia no tratamento do câncer de próstata e na recuperação da funcionalidade, com vídeos explicativos, relatos de pacientes e orientações de profissionais de saúde, incorporando recursos de IA.
- **OrthoTest** — projeto em desenvolvimento, que também contará com recursos de IA. *(link em breve)*

> Os conteúdos desses projetos têm caráter informativo e não substituem a avaliação de profissionais de saúde.

### 📓 Notebooks de estudo

A pasta [Notebooks](./Notebooks/) reúne experimentos e materiais de aprendizado, incluindo:

- **Ciência de Dados com Pandas** — manipulação e exploração de dados tabulares.
- **Aula de Data Science — OCEAN** — notebook de estudo aplicado a dados.
- **Segurança do paciente** — análise exploratória relacionada ao tema.
- **Treinamento de IA** — experimentação com técnicas de aprendizado de máquina.
- **RNA — Regressão** — estudos com redes neurais artificiais e regressão.

### 💬 Chatbot para Telegram e NLP

O notebook [telegram_chatbot_with_ia_2025.ipynb](./telegram_chatbot_with_ia_2025.ipynb) explora a construção de um chatbot com Python e Telegram, incluindo pré-processamento de texto em português, remoção de pontuação e acentos, tokenização, stopwords, vetorização TF-IDF e similaridade do cosseno para encontrar perguntas semelhantes.

## 🧰 Tecnologias e conceitos

- **Linguagem:** Python
- **IA generativa e orquestração:** LangChain, modelos da OpenAI
- **RAG e busca vetorial:** FAISS, embeddings, recuperação semântica
- **Processamento de documentos:** PyPDF
- **Interface web:** Streamlit
- **Machine Learning e NLP:** scikit-learn, TF-IDF, similaridade do cosseno, NLTK, Sentence Transformers
- **Análise de dados:** Pandas
- **Ambiente de experimentação:** Jupyter Notebook e Google Colab

Cada projeto possui seu próprio conjunto de dependências. Consulte o arquivo `requirements.txt` dentro da pasta correspondente antes de executar uma aplicação.

## 🚀 Como executar os assistentes RAG

### Pré-requisitos

- Python instalado.
- Git (opcional, para clonar o repositório).
- Uma chave de API da OpenAI para a versão que utiliza os serviços da OpenAI.
- Para a versão **Assistente RAG — Colab**, uma sessão do Google Colab com a API do modelo em execução e a URL correspondente configurada no código.

### 1. Clone o repositório

```bash
git clone https://github.com/alveslim/AI-Engineering.git
cd AI-Engineering
```

### 2. Escolha o projeto

**Versão com integração ao Google Colab:**

```bash
cd "Assistente RAG - Colab"
python -m venv .venv
```

Ative o ambiente virtual:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Instale as dependências e inicie a interface:

```bash
python -m pip install -r requirements.txt
streamlit run main.py
```

Antes de iniciar, confira a variável `COLAB_API_URL` em `main.py` e substitua-a pela URL válida da API que estiver rodando no Colab.

**Versão com API da OpenAI:**

```bash
cd "Assistente RAG - Politicas Internas"
python -m venv .venv
```

Ative o ambiente virtual conforme seu sistema operacional e instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Configure a variável de ambiente `OPENAI_API_KEY` em um arquivo `.env` dentro dessa pasta:

```env
OPENAI_API_KEY=sua_chave_aqui
```

Inicie a aplicação:

```bash
streamlit run app_basic.py
```

> **Segurança:** não compartilhe nem versione chaves de API, tokens ou outras credenciais. Mantenha o arquivo `.env` fora do Git.

## ▶️ Executar os notebooks

Os arquivos `.ipynb` podem ser abertos localmente com Jupyter Notebook/JupyterLab ou no Google Colab. Alguns notebooks dependem de datasets, downloads ou bibliotecas adicionais; confira as primeiras células e as instruções do próprio notebook antes de executar.

## 🗺️ Próximos passos

- Padronizar a documentação e as dependências de cada projeto.
- Adicionar exemplos de perguntas e respostas e critérios de avaliação dos assistentes RAG.
- Documentar configurações, limitações e procedimentos de execução de cada experimento.

## 👨‍💻 Autor

**Flávio Alves de Lima Neto**

- GitHub: [@alveslim](https://github.com/alveslim)

---

Este repositório está em evolução e reúne estudos e implementações práticas. A estrutura, as dependências e o nível de maturidade variam de um projeto para outro.
