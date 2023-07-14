import streamlit as st


def main():
    st.set_page_config(
        page_title="Eric - the PDF Oracle", page_icon=":books:", layout="wide"
    )
    st.header(":books: Eric - the PDF Oracle")
    st.text_input("Ask a question about your documents")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader('Upload your PDF\'s here and click on "Process"')
        st.button("Process")


if __name__ == "__main__":
    main()
