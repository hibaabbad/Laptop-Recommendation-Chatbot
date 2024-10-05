import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# Streamlit setup
st.title("Laptop Recommendation Chatbot")

# Cache for loading data
@st.cache_data
def load_data(file_path):
    # Load the document
    loader = CSVLoader(file_path, encoding='ISO-8859-1')
    return loader.load()

# Cache for splitting documents (prepend underscore to make it non-hashable)
@st.cache_data
def split_documents(_data):
    # Split the data into chunks
    splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', ': '], chunk_size=300, chunk_overlap=30)
    return splitter.split_documents(_data)

# Cache for loading embeddings
@st.cache_resource
def load_embeddings(model_name):
    # Load the embedding model
    return HuggingFaceEmbeddings(model_name=model_name)

# Cache for initializing the Groq model
@st.cache_resource
def init_groq_model(api_key, model_name):
    return ChatGroq(groq_api_key=api_key, model_name=model_name)

# Cache for setting up the vector store
@st.cache_resource
def setup_vector_store(_docs, _embeddings):
    return Chroma.from_documents(documents=_docs, embedding=_embeddings)

# Load data
data = load_data(os.getenv("csv_file_path"))
docs = split_documents(data)

# Load embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = load_embeddings(embedding_model_name)

# Load the Groq model
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key not set")

llm_groq = init_groq_model(api_key, 'llama3-8b-8192')

# Set up Chroma vector store and retriever
vectorstore = setup_vector_store(docs, embeddings)  # Now using _docs to avoid hash error
retriever = vectorstore.as_retriever()

# History aware retriever setup
contextualize_q_system_prompt = (
    "Given a chat history where a user is looking for a laptop based on specific requirements, "
    "formulate a standalone question that clearly identifies the user's laptop needs without referring "
    "to previous chat messages. The goal is to ensure the question can be understood and processed "
    "independently. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm_groq, retriever, contextualize_q_prompt)

# Question-answer chain
system_prompt = (
    "You are an assistant specialized in helping users find the best laptop from the available data based on their requirements. "
    "Use the provided laptop specifications and user preferences to recommend the most suitable options. "
    "If the user's requirements cannot be fully met, suggest the closest alternatives available. "
    "Keep your answers concise and focused on the laptop's key features such as price, performance, battery life, and portability. "
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm_groq, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# In-memory storage for chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

session_id = "user_session"
user_question = st.text_input("Ask a question about the best laptop for your needs:")

if user_question:
    answer = conversational_rag_chain.invoke(
        {"input": user_question},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    
    st.write("Answer:", answer)
