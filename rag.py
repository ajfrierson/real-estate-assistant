import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from uuid_utils import uuid4
from prompt import PROMPT, EXAMPLE_PROMPT

load_dotenv()

# Constants
CHUNK_SIZE = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """
    yield "Initialize components...✅"
    initialize_components()

    vector_store.reset_collection()

    yield "Load data...✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    yield "Split text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Add chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")
    
    qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",
                                      prompt=PROMPT,
                                      document_prompt=EXAMPLE_PROMPT)
    chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=vector_store.as_retriever(),
                                        reduce_k_below_max_tokens=True, max_tokens_limit=8000,
                                        return_source_documents=True)
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources_docs = [doc.metadata['source'] for doc in result['source_documents']]
    
    return result["answer"], sources_docs

if __name__ == '__main__':
    urls = [
        "https://www.wsj.com/buyside/personal-finance/mortgage/mortgage-rates-today-4-2-2026?gaa_at=eafs&gaa_n=AWEtsqdEbDlhhQ-ju_ht8dAjgmJYwDDLa2ikVQd1aMyKwHjj9lgO68bVpXOdegHMNkk%3D&gaa_ts=69cec35d&gaa_sig=VvkeRQQb1ZG4K-GxVNxftAO6S6xS_XdNWm-VnQewRycYlTsveDErZzxYacJwiGQqUuXvntI0h6RydLfufyzfaw%3D%3D",
        "https://finance.yahoo.com/personal-finance/mortgages/article/mortgage-rates-rise-for-5th-straight-week-above-6-mortgage-and-refinance-interest-rates-today-100000623.html"
    ]

    process_urls(urls)
    answer, sources = generate_answer("30 year mortgage rate")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")