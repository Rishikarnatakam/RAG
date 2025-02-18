import threading
import os
import faiss
import pickle
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.*")


# FastAPI Setup
app = FastAPI()

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text):
    """Cleans extracted text by removing unwanted characters."""
    return text.replace("\n", " ").strip()

import io

def process_pdf(pdf_file):
    """
    Reads and indexes the uploaded PDF.
    Args:
        pdf_file: Can be either bytes or a file-like object
    """
    # Convert to BytesIO if bytes were passed
    if isinstance(pdf_file, bytes):
        pdf_file = io.BytesIO(pdf_file)
    
    reader = PdfReader(pdf_file)
    chunks, metadata = [], []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned_text = clean_text(text)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100)
            page_chunks = text_splitter.split_text(cleaned_text)
            chunks.extend(page_chunks)
            metadata.extend([(chunk, page_num + 1) for chunk in page_chunks])
    
    if not chunks:
        raise ValueError("No text could be extracted from the PDF")
        
    print(f"Extracted {len(chunks)} chunks from PDF")
    
    embeddings = embedding_model.embed_documents([chunk for chunk, _ in metadata])
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings, dtype=np.float32))
    
    return index, metadata


def query_faiss(query, index, metadata, top_k=3):
    """Retrieves the top-k relevant text chunks."""
    query_embedding = np.array([embedding_model.embed_query(query)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return [(metadata[i], distances[0][idx]) for idx, i in enumerate(indices[0]) if i < len(metadata)]

def ask_gemini(query, context):
    """Queries Gemini with retrieved context."""
    prompt = f"""
    You are an AI assistant that answers questions based on retrieved document excerpts and add other things if they are suitable to the context:
    {context}
    **Question:** {query}
    
    Please provide a well-structured response and include references to page numbers.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
            "response_mime_type": "text/plain",
        },
    )
    response = model.generate_content(prompt)
    return response.text if response else "No response."

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(request: Request, file: UploadFile = File(...), question: str = Form(...)):
    """API endpoint to process PDF and get a response for a given question."""
    try:
        # Log request details
        logger.info(f"Received request headers: {request.headers}")
        logger.info(f"File name: {file.filename}")
        logger.info(f"Question: {question}")
        
        # Validate file
        if not file:
            logger.error("No file provided")
            return JSONResponse(content={"detail": "No file provided"}, status_code=400)
        
        if not file.filename.endswith('.pdf'):
            logger.error("Invalid file type")
            return JSONResponse(content={"detail": "Only PDF files are supported"}, status_code=400)
        
        # Read file contents
        try:
            contents = await file.read()
            logger.info(f"Successfully read file: {len(contents)} bytes")
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return JSONResponse(content={"detail": f"Error reading file: {str(e)}"}, status_code=400)
        
        # Process PDF
        try:
            index, metadata = process_pdf(contents)
            logger.info("PDF processed successfully")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return JSONResponse(content={"detail": f"Error processing PDF: {str(e)}"}, status_code=400)
        
        # Rest of your processing code...
        results = query_faiss(question, index, metadata)
        context = "\n".join([f"(Page {page}) {chunk}" for (chunk, page), _ in results])
        response = ask_gemini(question, context)
        
        return JSONResponse(content={
            "response": response,
            "retrieved_chunks": [
                {"page": page, "chunk": chunk, "score": float(score)}
                for (chunk, page), score in results
            ],
            "average_score": float(np.mean([score for _, score in results]))
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

# Add test endpoint
@app.get("/test")
async def test():
    return {"message": "API is working"}

def run_fastapi():
    """Function to run FastAPI in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Start FastAPI in a background thread
threading.Thread(target=run_fastapi, daemon=True).start()

# Streamlit Setup
st.set_page_config(page_title="RAG QA System", layout="wide", initial_sidebar_state="expanded")
st.title("📄 RAG-based Q&A System")
st.sidebar.header("Settings")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        index, metadata = process_pdf(uploaded_file)
    st.success("✅ PDF Indexed!")
    
    query = st.text_input("Ask a question:")
    
    if st.button("Search") and query:
        with st.spinner("Retrieving information..."):
            results = query_faiss(query, index, metadata)
            context = "\n".join([f"(Page {page}) {chunk}" for (chunk, page), _ in results])
            response = ask_gemini(query, context)
        
        # Metrics: Average score from retrieved chunks
        avg_score = np.mean([score for _, score in results])
        
        # Display Retrieved Chunks
        st.subheader("🔍 Retrieved Chunks:")
        for (chunk, page), score in results:
            st.markdown(f"**Page {page}**: {chunk} (Score: {score:.2f})")
        
        st.subheader("📝 Gemini Response:")
        st.write(response)
        
        # Show retrieval performance metric (average score)
        st.subheader("📊 Retrieval Metrics:")
        st.write(f"Average Score of Top-K Retrieved Chunks: {avg_score:.2f}")
