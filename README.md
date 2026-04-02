🏠 Real Estate Research Assistant
An AI-powered research tool that lets you extract and query information from real estate websites using Retrieval-Augmented Generation (RAG). Paste in URLs from any real estate or mortgage news source, and ask questions in plain English.

Features

🔗 URL Ingestion — Load up to 3 real estate or mortgage-related URLs at once
🧠 RAG Pipeline — Chunks, embeds, and stores content in a local vector database
💬 Natural Language Q&A — Ask questions and get AI-generated answers with source attribution
⚡ Fast Embeddings — Uses sentence-transformers/all-MiniLM-L6-v2 locally (no API key needed)
🦙 LLM via Groq — Powered by llama-3.3-70b-versatile for fast, high-quality answers


Tech Stack
ComponentTechnologyUIStreamlitLLMGroq (llama-3.3-70b-versatile)EmbeddingsHuggingFace (all-MiniLM-L6-v2)Vector StoreChroma (persisted locally)Document LoadingLangChain WebBaseLoaderText SplittingLangChain RecursiveCharacterTextSplitter

Project Structure
real_estate_assistant/
├── main.py                  # Streamlit UI
├── rag.py                   # RAG pipeline (load, embed, query)
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed)
└── resources/
    └── vectorstore/         # Persisted Chroma vector database

Setup
1. Clone the repository
bashgit clone https://github.com/your-username/real-estate-assistant.git
cd real-estate-assistant
2. Create and activate a virtual environment
bashpython -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Set up environment variables
Create a .env file in the project root:
envGROQ_API_KEY=your_groq_api_key_here
Get your free Groq API key at console.groq.com.

Running the App
bashstreamlit run main.py
Then open http://localhost:8501 in your browser.

How to Use

Enter URLs — Paste up to 3 real estate or mortgage-related URLs in the sidebar
Click "Process URLs" — The app will scrape, chunk, and index the content
Ask a question — Type a question like "What is the current 30-year mortgage rate?"
View the answer — The app displays the AI-generated answer along with the source URLs


Example Questions

What is the current 30-year fixed mortgage rate?
How have mortgage applications changed recently?
What are analysts predicting for home prices?
Is now a good time to buy or refinance?


Environment Variables
VariableDescriptionGROQ_API_KEYRequired. Your Groq API key for LLM inferenceHF_TOKENOptional. HuggingFace token for higher rate limits on model downloads

Notes

The vector store is persisted locally in resources/vectorstore/. It is reset each time you click Process URLs.
The embedding model (all-MiniLM-L6-v2) is downloaded automatically on first run and cached locally.
Set TOKENIZERS_PARALLELISM=false is handled automatically in rag.py to prevent tokenizer warnings.