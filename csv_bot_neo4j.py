import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_neo4j import Neo4jGraph,GraphCypherQAChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import GraphQAChain
from langchain_neo4j import Neo4jVector
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv(".env")
cohere_api_key = os.getenv("COHERE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Create llm object
llm = ChatCohere(model="command-r-03-2024",cohere_api_key=cohere_api_key)

# Create the vector index
neo4j_graph_vector_index = Neo4jVector.from_existing_graph(
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="Account_index",
    node_label="Account",
    text_node_properties=["website","size","account_name","phone_number", "location","industry"],
    embedding_node_property="embedding",
)

# Perform a similarity search
result = neo4j_graph_vector_index.similarity_search("What is the size of Castillo and Sons company", top_k=1)

# Assuming 'result' is your list of Document objects
# for doc in result:
    # Split the page_content by newline and iterate over each line
    # cleaned_page_content = dict(line.split(": ") for line in doc.page_content.strip().split('\n'))
    # print(f"""
    # website:{cleaned_page_content.get("website")}
    # size:{cleaned_page_content.get("size")}
    # account_name:{cleaned_page_content.get("account_name")}
    # phone_number:{cleaned_page_content.get("phone_number")}
    # location:{cleaned_page_content.get("location")}
    # industry:{cleaned_page_content.get("industry")}
    # """)


# # Create the retriever
# retriever=neo4j_graph_vector_index.as_retriever()


# Create the system prompt
system_prompt = ("""
Use the given context to answer the question thoroughly and accurately. 
If you don't know the answer, clearly state that you don't know. 
Provide detailed responses while keeping the explanation clear and concise, using up to six sentences. 
Ensure your answers are informative and directly relevant to the user's query. 

Context: {context}
"""
)

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

messages = [system_prompt, human_prompt]

qa_prompt = ChatPromptTemplate(
    messages=messages,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=neo4j_graph_vector_index.as_retriever(),
    chain_type="stuff",
)

qa_chain.combine_documents_chain.llm_chain.prompt = qa_prompt

# response = qa_chain.invoke("What is the phone number and website for Young-Richard?")

# print(response.get("result"))



graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

print(graph.schema)

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
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH c as customer, o.orderID as order_id).
If you need to divide numbers, make sure to
filter the denominator to be non-zero.

Examples:
# Retrieve the size of the account Myers, Hoffman and Lozano.
MATCH (a:Account {name: "Myers, Hoffman and Lozano"})
RETURN a.size AS AccountSize
# Find out the lead status for the account Myers, Hoffman and Lozano.
MATCH (a:Account {name: "Myers, Hoffman and Lozano"})-[:HAS_LEAD]->(l:Lead)
RETURN l.name AS LeadName, l.status AS LeadStatus
# Find the names of people whose task status is "In Progress".
MATCH (t:Task {status: "In Progress"})-[:ASSIGNED_TO]->(p:Person)
RETURN p.name AS PersonName, t.name AS TaskName
String category values:
Use existing strings and values from the schema provided. 

The question is:
{question}
"""
cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=cypher_generation_template
)




