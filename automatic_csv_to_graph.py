import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from neo4j_runway import (Discovery, GraphDataModeler, 
                          PyIngest,UserInput)
from neo4j_runway.code_generation import PyIngestConfigGenerator
from neo4j_runway.llm.openai import OpenAIDataModelingLLM, OpenAIDiscoveryLLM
from neo4j_runway.utils import test_database_connection
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")


def main():
    st.header("Chat with you csv file")
    "This is a straightforward chat application designed to allow users to ask questions about the data contained within a CSV file."
    "Make sure the data present in the csv file is pre-processed."
    csv_file = st.file_uploader("Upload your csv file", type="csv")
    if csv_file:
        data = pd.read_csv(csv_file)
        st.dataframe(data=data.head())
        "Please provide USER_GENERATED_INPUT"
        "The USER_GENERATED_INPUT variable contains a general discription and feature descriptions for each feature we'd like to use in our graph based on the csv data."
        "General description example:- This is data on different countries."
        general_description = st.text_input("Enter a general description")
        "Column description should be given in key: value pair followed by commas if there are mulitple keys."
        """Example:- 'id': 'unique id for a country',
                    'name': 'the country name', 
                    phone_code': 'country area code"""
        column_descriptions = dict(st.text_area("Enter column descriptions"))
        "Use cases are questions that we would ask based on the data present in the csv"
        "Input questions separated by commas"
        use_cases = st.text_area("Enter use cases (atleast 3)")

        neo4j_uri = st.text_input("Enter your Neo4j URI")
        neo4j_username = st.text_input("Enter your Neo4j username")
        neo4j_password = st.text_input("Enter your Neo4j password")


        USER_GENERATED_INPUT = UserInput(general_description=general_description,
                                 column_descriptions=column_descriptions,
                                use_cases=[use_cases])

        
        llm_disc = OpenAIDiscoveryLLM(model_name='o1-mini-2024-09-12', model_params={"temperature": 0})
        llm_dm = OpenAIDataModelingLLM(model_name='o1-mini-2024-09-12', model_params={"temperature": 0.5})

        disc = Discovery(llm=llm_disc, user_input=USER_GENERATED_INPUT, data=data)
        disc.run(show_result=True, notebook=True)
        
        
        

    
if __name__ == "__main__":
    main()