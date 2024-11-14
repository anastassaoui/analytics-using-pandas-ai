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

# Fetch API keys from .env file
groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key_helper = os.getenv("GROQ_API_KEY_helper")

st.set_page_config(page_title="Absence Management Dashboard", layout="wide")

# Initialize the primary model for PandasAI processing
try:
    pandas_ai_model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize the PandasAI model: {e}")
    st.stop()

# Initialize the helper model for prompt reformulation using a different API key
try:
    helper_model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key_helper)
except Exception as e:
    st.error(f"Failed to initialize the helper model: {e}")
    st.stop()

st.title("Data Analysis with PandasAI")

# File uploader to accept CSV and XLSX files
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Here are the first few rows of your data:")
        st.write(data.head(150))
        
        # SmartDataframe with PandasAI model
        df = SmartDataframe(data, config={"llm": pandas_ai_model})
        
        # Function to reformulate user prompt for PandasAI
        def reformulate_prompt(user_prompt, data):
            column_info = data.columns.tolist()
            sample_data = data.head(5).to_dict()
            context_prompt = (
                f"Given the following columns: {column_info} "
                f"and the first five rows of data: {sample_data}, "
                f"reformulate the user's question for data in a way the pandasAi would understand it and dont say anything elese other then the prompt pandas Ai would comprehend: '{user_prompt}'"
            )
            try:
                # Use call_as_llm if available to send prompt directly
                response = helper_model.call_as_llm(context_prompt)
                return response
            except Exception as e:
                st.error(f"Error in prompt reformulation: {e}")
                return None


        # Text area for input prompt
        prompt = st.text_area("Enter your prompt:")

        if st.button("Generate"):
            if prompt:
                # Reformulate the prompt before passing to PandasAI
                with st.spinner("Reformulating prompt..."):
                    reformulated_prompt = reformulate_prompt(prompt, data)
                
                if reformulated_prompt:
                    with st.spinner("Generating response with PandasAI..."):
                        try:
                            # Generate response using the reformulated prompt
                            response = df.chat(reformulated_prompt)
                            
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
                            st.error(f"Failed to generate response with PandasAI: {e}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
