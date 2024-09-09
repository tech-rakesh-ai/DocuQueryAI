# DocuQuery AI

**DocuQuery AI** is an intelligent document query application built using Streamlit. It allows users to upload PDF and DOCX files, processes them into vector embeddings for efficient document retrieval, and then enables users to query the contents of the documents using advanced language models. The application supports several pre-trained large language models and provides a user-friendly interface to query document content efficiently.

## Features

### 1. **File Uploading**
- Users can upload multiple files in PDF or DOCX format.
- The application processes and extracts the content of the files to be used in vector embeddings for easy document retrieval.

### 2. **Document Vectorization**
- The uploaded documents are split into smaller chunks and converted into vector embeddings using the `GoogleGenerativeAIEmbeddings`.
- The FAISS vector store is used to store and manage the vectorized document chunks.

### 3. **Model Selection**
- Users can choose from a variety of powerful language models available in the sidebar. Models include:
  - `gemma-7b-it`
  - `gemma2-9b-it`
  - `mixtral-8x7b-32768`
  - `llama3-70b-8192`
  - `llama3-8B`
  - ...and more.
  
### 4. **Natural Language Query**
- After processing the documents, users can ask questions in natural language. The application retrieves the relevant information from the uploaded documents and returns the most accurate response based on the context.
  
### 5. **Response Time and Model Info**
- The application displays the response time and the selected model for each query.
  
### 6. **Document Similarity Search**
- The retrieved response includes document chunks that are most similar to the user's query, displayed in the expanded view for in-depth review.

### 7. **User-Friendly Interface**
- The app uses Streamlit to offer an intuitive and interactive experience, making it easy to upload documents, ask questions, and explore results.

### 8. **Session Management and Temporary File Cleanup**
- Each session automatically cleans up temporary files used during document upload and vectorization to optimize space and performance.

## Installation and Setup

Follow the steps below to install and run DocuQuery AI on your local machine.

### Prerequisites
- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-repository/docuquery-ai.git
cd docuquery-ai
```

### Step 2: Create and Activate a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Step 3: Install the Required Dependencies

```bash
pip install -r requirements.txt
```

Make sure your `requirements.txt` contains the following dependencies:

```text
streamlit
langchain
langchain_groq
langchain_google_genai
langchain_community
faiss-cpu
PyPDF2
docx
shutil
```

### Step 4: Set Up API Keys

Create a file called `.streamlit/secrets.toml` in the root directory of your project and add your API keys:

```toml
[secrets]
GROQ_API_KEY = "your-groq-api-key"
GOOGLE_API_KEY = "your-google-api-key"
```

Replace `your-groq-api-key` and `your-google-api-key` with your actual API keys for the respective services.

### Step 5: Run the Application

Start the Streamlit application by running the following command:

```bash
streamlit run app.py
```

The application will open in your browser, allowing you to upload documents and start querying them.

### Step 6: Usage Guide

1. **Upload Documents**: On the sidebar, upload your PDF or DOCX files. You can upload multiple files at once.
2. **Process Files**: Click the "Upload & Process Files" button to process your files. This will convert the document content into vector embeddings.
3. **Ask Questions**: Once the files are processed, use the text input at the bottom to ask any questions related to the content of the uploaded documents.
4. **Review Results**: The results will include the AI's answer and a list of document chunks that match your query the most.

## Application Structure

```bash
docuquery-ai/
│
├── app.py                     # Main application code
├── requirements.txt            # Dependencies file
└── .streamlit/
    └── secrets.toml            # API keys (not included in the repository)
```

## Additional Information

- **Document Vectorization**: After documents are uploaded, they are split into chunks using `RecursiveCharacterTextSplitter`. These chunks are embedded into vectors using Google Generative AI embeddings and stored using the FAISS vector store.
- **Document Retrieval**: When a user inputs a query, the system retrieves the most relevant document chunks using FAISS and responds based on the content retrieved.
- **Model Switching**: Users can switch between different pre-trained language models in the sidebar to get responses tailored by different architectures.

## Future Enhancements
- Integration with additional file formats (e.g., TXT, HTML).
- Enhanced document visualization and interaction features.
- Real-time document summarization and preview before query.
  
## Author

Developed by **Rakesh Kumar**.  
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/m-rakesh-kr/) and share your feedback!

## License

MIT License © 2024 DocuQuery AI.
