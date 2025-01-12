import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import ChatCohere

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
cohere_api_key = os.getenv("COHERE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

logger = get_logger(__name__)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key)


llm = ChatCohere(model="command-r-03-2024",cohere_api_key=cohere_api_key)


def main():
    st.header("📄Chat with your pdf file")

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Store the chunks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url="neo4j+s://27c34e40.databases.neo4j.io",
            username=username,
            password="GCFJxud3E61HTHDETNWh0pjXzQWf8jwx_thonEmHG7g",
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=True,  # Delete existing PDF data
        )
        retriever=vectorstore.as_retriever()
        
        system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file")

        if query:
            response = chain.invoke({"input": query})
            st.write(response["answer"])


if __name__ == "__main__":
    main()