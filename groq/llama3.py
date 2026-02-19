import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv

load_dotenv()

# Load groq api key
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI

st.title("Chatgroq with llama3 model")

# Define model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key = groq_api_key
)

# Prompt template

prompt = PromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embeddings():

    if 'vectors' not in st.session_state:

        st.session_state.embeddings=OllamaEmbeddings(model = "llama2")
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Ollama embeddings

prompt1 = st.text_input("Enter your question from documents")

if st.button("Document Embedding"):
    vector_embeddings()
    st.write("Vector store DB is readyyy.")




if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt1})
    print("response time: ",time.process_time()-start)
    st.write(response['answer'])

    # with a streamlit expander
    with st.expander("Document Similarity Search"):
    # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
