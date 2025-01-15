import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_neo4j import Neo4jGraph,GraphCypherQAChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Environment variables
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
cohere_api_key = os.getenv("COHERE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

logger = get_logger(__name__)

# Initialize embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
llm = ChatCohere(
    model="command-r-03-2024",
    cohere_api_key=cohere_api_key
)

def process_csv_file():

    # Create LLM object
    llm = ChatCohere(
        model="command-r-03-2024",
        cohere_api_key=cohere_api_key,
        temperature=0
    )

    # Create the vector index
    neo4j_graph_vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        index_name="Account_index",
        node_label="Account",
        text_node_properties=["website", "size", "account_name", "phone_number", "location", "industry"],
        embedding_node_property="embedding",
    )

    # Define system prompt
    system_prompt = """
    Use the given context to answer the question thoroughly and accurately. 
    If you don't know the answer, clearly state that you don't know. 
    Provide detailed responses while keeping the explanation clear and concise, using up to six sentences. 
    Ensure your answers are informative and directly relevant to the user's query. 
    Context: {context}
    """
    chat_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=system_prompt
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="Can you provide details on: {question}?"
        )
    )

    qa_prompt = ChatPromptTemplate(
        messages=[chat_system_prompt, human_prompt],
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=neo4j_graph_vector_index.as_retriever(),
        chain_type="stuff",
    )

    qa_chain.combine_documents_chain.llm_chain.prompt = qa_prompt

    # Cypher QA Chain
    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )

    cypher_generation_template = """
    Task:
    Generate Cypher query for a Neo4j graph database.

    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.

    Schema:
    {schema}

    Note:
    Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything other than
    for you to construct a Cypher statement.
    The question is:
    {question}
    """
    cypher_generation_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=cypher_generation_template
    )

    qa_generation_template_str = """
    You are an assistant that takes the results from a Neo4j Cypher query and forms a human-readable response. The query results section contains the results of a Cypher query that was generated based on a user's natural language question. The provided information is authoritative; you must never question it or use your internal knowledge to alter it. 
    Query Results:
    {context}
    Question:
    {question}
    Helpful Answer:
    """
    qa_generation_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_generation_template_str
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        top_k=10,
        allow_dangerous_requests=True,
        graph=graph,
        verbose=True,
        validate_cypher=True,
        qa_prompt=qa_generation_prompt,
        cypher_prompt=cypher_generation_prompt,
        qa_llm=llm,
        cypher_llm=ChatCohere(model="command-r-plus-04-2024", cohere_api_key=cohere_api_key, temperature=0),
    )

    # Accept user queries
    query = st.text_input("Ask questions about your CRM data")
    if query:
        response = cypher_chain.invoke(query)
        st.write(response["result"])

    

def process_pdf_file(uploaded_file):
    st.write("PDF uploaded successfully")
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Langchain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Store chunks in Neo4j vector database
    vectorstore = Neo4jVector.from_texts(
        chunks,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name="pdf_bot",
        node_label="PdfBotChunk",
        pre_delete_collection=True  # Delete existing PDF data
    )
    retriever = vectorstore.as_retriever()

    # Define system prompt
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    # Accept user queries
    query = st.text_input("Ask questions about your PDF file")
    if query:
        response = chain.invoke({"input": query})
        st.write(response["answer"])

def main():
    st.header("📄 Chat with your PDF or CSV file")
    uploaded_file = st.file_uploader("Upload your PDF or CSV", type=["pdf", "csv"])

    if uploaded_file is None:
            process_csv_file()
    else:
        process_pdf_file(uploaded_file)

if __name__ == "__main__":
    main()
