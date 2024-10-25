import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image
import io

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
                        # Generate response
                        response = df.chat(prompt)
                        
                        # Check if response is a saved chart path, load it and display
                        if isinstance(response, str) and "temp_chart.png" in response:
                            image = Image.open(response)  # Open the image from the saved path
                            st.image(image)  # Display the image in Streamlit
                            
                            # Convert image to bytes for download
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            # Add download button
                            st.download_button(
                                label="Download Chart",
                                data=img_byte_arr,
                                file_name="generated_chart.png",
                                mime="image/png"
                            )
                        else:
                            st.write(response)
                    except Exception as e:
                        st.error(f"Failed to generate response: {e}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
