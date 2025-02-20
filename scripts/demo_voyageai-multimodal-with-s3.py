import logging
import openai
from typing import List
from MultiModalRetriever import MultiModalRetriever
import voyageai

# load .ENV file
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import voyageai
voyageai_api_key = "<VOYAGEAI_API_KEY>"
MODEL_NAME = "voyage-multimodal-3"
vo = voyageai.Client(api_key=voyageai_api_key)

# Setup S3
import boto3
access_key = '<AWS_ACCESS_KEY>'
secret_key = '<AWS_SECRET_KEY>'
session_token = '<AWS_SESSION_TOKEN>'
s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, aws_session_token=session_token)


# Get Embedding Function
def get_embedding(text: any) -> List[float]:
    try:
        # Embed the documents
        return vo.multimodal_embed(
            [[text]], model=MODEL_NAME, input_type="document"
        ).embeddings[0]
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

# Example usage
from mdb_toolkit import CustomMongoClient
print("mdb_toolkit package imported successfully")

# Define database and collection names
database_name = "test_database"
collection_name = "test_collection"
index_name = "vs_1"  # Ensure this matches your intended index name
distance_metric = "cosine"

client = CustomMongoClient(
    "mongodb://localhost:27017/?directConnection=true&serverSelectionTimeoutMS=2000",
    get_embedding=get_embedding
)

# Create the MultiModalRetriever instance
retriever = MultiModalRetriever(
    mongo_client=client,
    database_name=database_name,
    collection_name=collection_name,
    index_name=index_name,
    s3_client=s3_client,
    bucket_name="test_bucket",
    voyage_api_key=voyageai_api_key
)

# Create the search index
client._create_search_index(
    database_name=database_name,
    collection_name=collection_name,
    index_name=index_name,
    distance_metric=distance_metric,
    embedding_field="content_embedding", #voyageai :)
)

# Wait for the search index to be READY
logger.info("Waiting for the search index to be READY...")
index_ready = client.wait_for_index_ready(
    database_name=database_name,
    collection_name=collection_name,
    index_name=index_name,
    max_attempts=10,
    wait_seconds=1
)

if index_ready:
    logger.info(f"Search index '{index_name}' is now READY and available!")
    print("Index is ready!")
else:
    logger.error("Index creation process exceeded wait limit or failed.")
    print("Index creation process exceeded wait limit.")
    exit()

pdfs = ["s3://multimodal-rag-test-jz/fdr-readingcopy.pdf"]


# Insert documents into the collection
retriever.mm_embed(pdfs)

print("Inserted documents into the collection.")

# insert pause to allow index update
import time
time.sleep(5)
print("Searching for documents...")

# Perform Multimodal search
# 1. Vector-Based Search
query = "The consequences of a dictator's peace"
logger.info(f"Performing vector-based search with query: '{query}'")

vector_results = retriever.mm_query(query, k=3)
print("\n--- Vector-Based Search Results ---")
for doc in vector_results:
    print(f"ID: {doc.get('_id')}\nS3 Path: {doc.get('s3_full_path')}\nScore: {doc.get('score')}\n")
