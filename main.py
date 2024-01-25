import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import os
st.set_page_config(layout="wide")
st.title("Answer IT")
st.sidebar.title("Document uploading section")


uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=['pdf'])
if uploaded_file:
    st.sidebar.info("Uploaded file")
    filepath = os.path.join("data", uploaded_file.name)  # Save the file in the "data" folder
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    with open(filepath, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

placeholder = st.empty()
checkpoint = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

query = st.text_input("Question: ")
process_text = st.button("Ask Question")
if process_text:

    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    prompt = f"""{final_texts}
    QUERY: {query}
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))

