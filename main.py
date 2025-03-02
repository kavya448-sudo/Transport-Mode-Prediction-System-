import os
import time
import streamlit as st
import fitz  # PyMuPDF to extract text from PDF
import openai  # OpenAI API for GPT models
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (OpenAI API key)
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

# Streamlit session state for vector embeddings
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit app title
st.title("Transport Mode Prediction Demo")

# Initialize OpenAI API
openai.api_key = openai_api_key

# Define the prompt template for context-based answering
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

document_chain = create_stuff_documents_chain(openai.Completion.create, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# PDF upload
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# Extract text from the uploaded PDF
if pdf_file:
    doc = fitz.open(pdf_file)
    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text()

    st.text_area("Extracted Text from PDF", pdf_text[:1500])  # Display first 1500 characters

# User input for address and price range
from_address = st.text_input("Enter From Address")
to_address = st.text_input("Enter To Address")
price_min = st.number_input("Enter Minimum Price", min_value=0, value=100)
price_max = st.number_input("Enter Maximum Price", min_value=0, value=500)

# Function to predict transport mode (based on some logic)
def predict_transport_mode(from_address, to_address, price_min, price_max):
    # This is a mock logic. You can use more complex rules or ML models here
    if price_max > 200 and "sea" in to_address.lower():
        return "Shipping"
    elif price_max < 200 and "airport" in to_address.lower():
        return "Air"
    else:
        return "Road"

# Determine transport mode after user provides input
if from_address and to_address:
    transport_mode = predict_transport_mode(from_address, to_address, price_min, price_max)
    st.write(f"The predicted transport mode is: {transport_mode}")

# Process prompt and display response
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()

    # Call OpenAI's GPT model to process the prompt (assuming GPT-3.5/4 or any other variant)
    response = openai.Completion.create(
        model="text-davinci-003",  # You can change to GPT-4 or other models if you have access
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7
    )

    # Display the response
    st.write("Response time:", time.process_time() - start)
    st.write(response['choices'][0]['text'].strip())

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")