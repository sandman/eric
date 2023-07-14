import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import yaml

with open("config.yml") as parameters:
    config = yaml.safe_load(parameters)

embedding_config = config["embedding"]
print(embedding_config)


def get_pdf_text(pdf_docs):
    raw_text = ""
    for doc in pdf_docs:
        pdf = PdfReader(doc)
        for page in pdf.pages:
            raw_text += page.extract_text()
    return raw_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks


def get_vectorstore(chunks):
    if embedding_config == "openai":
        embeddings = OpenAIEmbeddings()
    if embedding_config == "instructor":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Eric - the PDF Oracle", page_icon=":books:", layout="wide"
    )
    st.header(":books: Eric - the PDF Oracle")
    st.text_input("Ask a question about your documents")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            'Upload your PDF\'s here and click on "Process"', accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks and embeddings
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)


if __name__ == "__main__":
    main()
