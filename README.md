RAG-based Q&A System - README
Introduction
This project implements a Retrieval-Augmented Generation (RAG) based Q&A system using a PDF document as a knowledge base. The system indexes the contents of the PDF, retrieves relevant chunks of text based on the user query, and then uses Gemini-2.0 to generate an answer based on the retrieved information.
Features
- Upload a PDF document to use as the knowledge base.
- Ask questions related to the content of the PDF.
- Retrieve the most relevant sections of the document.
- Generate answers using Gemini-2.0 model.
- Display relevant context along with the answer.
Installation
To run this system locally, follow the steps below to set up the environment and install necessary dependencies:
1. Clone the repository:
   ```
   git clone <https://github.com/Rishikarnatakam/RAG.git>
   cd <RAG>
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment variables for the Gemini API key:
   Create a `.env` file and add your Gemini API key as:
   ```
   GEMINI_API_KEY=<your_api_key_here>
   ```
4. Start the FastAPI server (in a separate terminal):
   ```
   python main.py
   ```
5. Launch the Streamlit interface (in another terminal):
   ```
   streamlit run app.py
   ```

Usage
1. Upload a PDF document using the file uploader in the Streamlit app.
2. Enter your question in the input box.
3. Click the 'Search' button to retrieve relevant document chunks and generate an answer.
API Documentation
This project exposes a FastAPI endpoint to interact with the backend. The API is used to process the PDF document and provide answers to queries.
The main endpoint is `/ask`:
- **POST** `/ask`: Takes a PDF file and a query, processes the PDF, and returns the response along with the retrieved chunks.
Example request:
   - Method: POST
   - URL: `http://localhost:8001/ask`
   - Body: A PDF file and a string with the query.
Example response:
   ```json
   {
       'response': 'The answer generated from Gemini model',
       'retrieved_chunks': [{ 'page': 1, 'chunk': 'Text from page 1', 'score': 0.95 }],
       'average_score': 0.93
   }
   ```

License
This project is licensed under the MIT License - see the LICENSE file for details.
