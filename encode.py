# https://www.mongodb.com/docs/atlas/atlas-search/knn-beta/
# Import necessary libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymongo
import requests
from pathlib import Path
import os
from PyPDF2 import PdfReader
from openai.embeddings_utils import get_embedding
import datetime
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 60
MONGO_URI = "<MONGO_URI>"
MONGODB_DATABASE = "llamatest"

# Define a function to split text into chunks
def text_to_chunks(text):
    """Splits text into chunks of the given size.

    Args:
        text: The text to split.
        chunk_size: The size of the chunks.

    Returns:
        A list of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.create_documents([text])

# Define a function to download a PDF file
def download_pdf(url, out_dir):
    """Downloads a PDF file from the given URL and saves it to the given directory.

    Args:
        url: The URL of the PDF file to download.
        out_dir: The directory to save the PDF file to.

    Returns:
        The path to the downloaded PDF file.
    """
    # Check if the directory exists
    if not out_dir.exists():
        os.makedirs(out_dir)

    # Get the path to the downloaded PDF file
    out_path = out_dir / "paper.pdf"

    # Check if the file already exists
    if not out_path.exists():
        # Download the file
        r = requests.get(url)
        with open(out_path, "wb") as f:
            f.write(r.content)

    # Return the path to the downloaded file
    return out_path

# Define a function to load data from a PDF file
def load_data(out_path):
    """Loads the data from the given PDF file.

    Args:
        out_path: The path to the PDF file to load data from.

    Returns:
        The data from the PDF file.
    """
    # Create a PDF reader
    loader = PdfReader(Path(out_path))

    # Return the data from the PDF file
    return loader

# Define a function to insert a batch of documents into a collection in MongoDB
def resetCollection(collection, batch):
    """Inserts a batch of documents into a collection in MongoDB.

    Args:
        collection: The collection to insert the documents into.
        batch: The batch of documents to insert.

    Returns:
        None
    """
    # Delete all documents from the collection
    collection.delete_many({})
    # Insert the batch of documents into the collection
    collection.insert_many(batch)

def main():
    PDF_URI = 'https://webassets.mongodb.com/_com_assets/cms/l07ypsi8jjr06oq7j-08%20Jan22%20WhP-The%205%20Phases%20of%20Banking%20Modernization2.pdf'
    out_dir = Path("data")
    out_path = download_pdf(PDF_URI, out_dir)
    doc = load_data(out_path)
    # Connect to the MongoDB server
    client = pymongo.MongoClient(MONGO_URI)
    # Get the collection
    collection = client.llamatest.vectorific
    
    batch = []
    for i,tmpPage in enumerate(doc.pages):
        tmpPageText = " ".join(tmpPage.extract_text().split()) #funky, but cleans up the \n and all that jazz
        tmpPageChunks = text_to_chunks(tmpPageText)
        for idx,tmpChunk in enumerate(tmpPageChunks):
            batch.append({
                "content":tmpChunk.page_content,
                "embeddings":get_embedding(tmpChunk.page_content,engine="text-search-ada-doc-001"),
                "meta":{
                    "chunk_number":"page:"+str(i)+"-"+str(idx),
                    "url":PDF_URI, 
                    "created_on": datetime.datetime.utcnow()
                }
            })
    resetCollection(collection,batch)
main()