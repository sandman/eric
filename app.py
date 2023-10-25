import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, user_template, bot_template
import yaml

with open("config.yml") as parameters:
    config = yaml.safe_load(parameters)

embedding_config = config["embedding"]
model_config = config["model"]


def get_pdf_text(pdf_docs):
    """Extract text from PDFs and return as a single string"""
    raw_text = ""
    for doc in pdf_docs:
        pdf = PdfReader(doc)
        for page in pdf.pages:
            raw_text += page.extract_text()
    return raw_text


def get_text_chunks(raw_text):
    """Split text into chunks of 1000 characters with 200 characters overlap"""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks


def get_vectorstore(chunks):
    """Create vector store from text chunks"""
    if embedding_config == "openai":
        embeddings = OpenAIEmbeddings()
    if embedding_config == "instructor":
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl"
        )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore


def handle_user_input(user_question):
    """Handle user input"""
    response = st.session_state.conversation({"question": user_question})
    st.write(response)


def get_conversation_chain(vectorstore):
    if model_config == "openai":
        llms = ChatOpenAI()
    elif model_config == "huggingface":
        llms = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llms, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Eric - the PDF chatbot", page_icon=":books:", layout="wide"
    )
    st.header(":books: Eric - the PDF chatbot")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_user_input(user_question)

    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.write(
        user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True
    )
    st.write(
        bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True
    )

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            'Upload your PDF\'s here and click on "Process"',
            accept_multiple_files=True,
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks and embeddings
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )

        # st.session_state.conversation


if __name__ == "__main__":
    main()
