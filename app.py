import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Fetch API key from .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLaMA3 model using Groq's API key
try:
    model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize the LLaMA model: {e}")
    st.stop()

st.title("Data Analysis with PandasAI")

# File uploader to accept CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Here are the first few rows of your data:")
        st.write(data.head(150))
        
        df = SmartDataframe(data, config={"llm": model})
        
        # Text area for input prompt
        prompt = st.text_area("Enter your prompt:")
        
        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    try:
                        # Using PandasAI to generate response based on the dataframe
                        response = df.chat(prompt)
                        st.write(response)
                    except Exception as e:
                        st.error(f"Failed to generate response: {e}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
