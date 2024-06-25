import os
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

# Image to Text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text

# LLM (Large Language Model)
def generate_description(scenario):
    template = """
    You are an image describer.
    You can generate a short description based on an image, but the description should be no more than 20 words.

    CONTEXT: {scenario}
    IMAGE DESCRIPTION:
    """

    API_KEY = os.getenv('OPENAI_API_KEY')
    description_llm = ChatOpenAI(openai_api_key=API_KEY)

    prompt = template.format(scenario=scenario)
    description = description_llm.predict(prompt)

    print(description)
    return description

# Text to Speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    if response.status_code == 200:
        with open('audio.flac', 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to generate speech: {response.status_code}, {response.text}")

# Main
def main():
    st.set_page_config(page_title="Image to Audio Description")
    st.header("Turn an image into an audio description")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()

        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        scenario = img2text(uploaded_file.name)
        description = generate_description(scenario)
        text2speech(description)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("description"):
            st.write(description)
        
        st.audio("audio.flac")

if __name__ == "__main__":
    main()
