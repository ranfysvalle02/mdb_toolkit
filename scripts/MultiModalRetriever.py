import numpy as np
from voyageai import Client
from abc import ABC, abstractmethod
import boto3
from botocore.exceptions import NoCredentialsError
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image


MODEL_NAME = "voyage-multimodal-3"

def upload_to_s3(local_file, s3_client, bucket_name, s3_key):
    try:
        s3_client.upload_file(local_file, bucket_name, s3_key)
        print(f"Upload Successful: {local_file} to s3://{bucket_name}/{s3_key}")
        return True
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


class InputHandler(ABC):
    @abstractmethod
    def preprocess(self, input):
        pass

    def parse_metadata(self, input):
        file_metadata = {}
        bucket_name = input.split("/")[2]
        s3_key = "/".join(input.split("/")[3:])

        file_metadata["s3_full_path"] = input
        file_metadata["s3_bucket_name"] = bucket_name
        file_metadata["s3_key"] = s3_key
        return file_metadata
        

class S3PDFHandler(InputHandler):

    def _pdf_to_screenshots(self, s3_client, bucket_name, s3_key, zoom=3.0):
        # open the PDF from S3 and extract the images
        pdf_data = s3_client.get_object(Bucket=bucket_name, Key=s3_key)["Body"].read()

        pdf_stream = BytesIO(pdf_data)
        pdf = fitz.open(stream=pdf_stream, filetype="pdf")

        images = []

        # Loop through each page, render as pixmap, and convert to PIL Image
        mat = fitz.Matrix(zoom, zoom)
        for n in range(pdf.page_count):
            pix = pdf[n].get_pixmap(matrix=mat)

            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        # Close the document
        pdf.close()
        return images
    

    def preprocess(self, s3_client, input):
        file_metadata = super().parse_metadata(input)
        images = self._pdf_to_screenshots(s3_client, file_metadata["s3_bucket_name"], file_metadata["s3_key"])
        
        return images, file_metadata


class S3ImageHandler(InputHandler):

    def _open_s3_image(self, s3_client, bucket_name, s3_key):
        image_data = s3_client.get_object(Bucket=bucket_name, Key=s3_key)["Body"].read()
        img = Image.open(BytesIO(image_data))
        return img
    
    def preprocess(self, s3_client, input):
        file_metadata = super().parse_metadata(input)
        image = self._open_s3_image(s3_client, file_metadata["s3_bucket_name"], file_metadata["s3_key"])
        
        return [image], file_metadata

class ImageHandler(S3ImageHandler):
    def __init__(self, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name

    def preprocess(self, s3_client, input):
        file_name = input.split("/")[-1]
        upload_to_s3(input, s3_client, self.bucket_name, file_name)
        return super().preprocess(s3_client, f"s3://{self.bucket_name}/{file_name}") 

class PDFHandler(S3PDFHandler):
    def __init__(self, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name

    def preprocess(self, s3_client, input):
        file_name = input.split("/")[-1]
        upload_to_s3(input, s3_client, self.bucket_name, file_name)
        return super().preprocess(s3_client, f"s3://{self.bucket_name}/{file_name}") 






class MultiModalRetriever():
    def __init__(self, mongo_client, database_name, collection_name, index_name, s3_client, bucket_name, voyage_api_key):
        self.client = mongo_client
        self.database_name = database_name
        self.collection_name = collection_name
        self.index_name = index_name
        self.s3 = s3_client
        self.bucket_name = bucket_name
        self.vo = self._get_voyage_client(voyage_api_key)


    def _get_voyage_client(self, voyage_api_key):
        return Client(api_key=voyage_api_key)


    def _create_input_processor(self, input):
        input_format = "pdf"
        if input.startswith("s3://"):
            if input.endswith(".pdf"):
                return S3PDFHandler()
            else:
                return S3ImageHandler()
        else: 
            if input.endswith(".pdf"):
                return PDFHandler(bucket_name=self.bucket_name)
            else:
                return ImageHandler(bucket_name=self.bucket_name)
    
    
    def _create_embedding(self, processed_inputs):
        # handle parsing out images from pdfs, and/or pulling images from s3
        # create and return the actual embeddings
        return np.array(
            self.vo.multimodal_embed(
            inputs=[[x] for x in processed_inputs],
            model=MODEL_NAME,
            input_type="document"
        ).embeddings)

    def _store_embedding(self, document_vectors, metadata):
        # store the embeddings in mongodb alongside the 
        # file metadata in S3
        for i, doc_vector in enumerate(document_vectors):
            self.client[self.database_name][self.collection_name].insert_one({
                "s3_full_path": metadata["s3_full_path"],
                "s3_bucket_name": metadata["s3_bucket_name"],
                "s3_key": metadata["s3_key"],
                "content_embedding": doc_vector.tolist()
            })
        return True

    def mm_embed(self, inputs):
        # main entry point for the multimodal retriever utility
        # establishes embeddings for the input files and saves them
        # to mongodb

        for input_file in inputs:
            handler = self._create_input_processor(input_file)
            processed_inputs, metadata = handler.preprocess(self.s3, input_file)
            document_vectors = self._create_embedding(processed_inputs)
            self._store_embedding(document_vectors, metadata)
        
        return True
        

    def mm_query(self, query, k=5):
        # query the multimodal retriever for the most similar
        # documents to the query
        vector_results = self.client.vector_search(
            query=query,
            limit=k,
            database_name=self.database_name,
            collection_name=self.collection_name,
            index_name=self.index_name,
            embedding_field="content_embedding", #voyageai :)
        )
        return vector_results


        
