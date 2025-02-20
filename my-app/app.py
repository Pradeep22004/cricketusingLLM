from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)

from flask_cors import CORS
CORS(app)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/content/drive/MyDrive/models/llama-2-7b-chat.Q8_0.gguf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load the model on the specified device (cuda)
model_llama = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # If you have enough GPU memory
    trust_remote_code=True,
    device_map="auto",  # Or specify device_map="cuda"
)

print("Model loaded successfully.")

from ctransformers import AutoModelForCausalLM
model_path = "/content/drive/MyDrive/models/llama-2-7b-chat.Q8_0.gguf"
model_llama = AutoModelForCausalLM.from_pretrained(model_path,max_new_tokens=4096,context_length=4096,model_type='llama',device='cuda')
print("Model loaded successfully.")

from pinecone import Pinecone
pc = Pinecone(api_key=userdata.get('pc_api_key'))

import time
from pinecone import ServerlessSpec
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = 'data1'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]
if index_name not in existing_indexes:
    pc.create_index(
        index_name,
        dimension=1024,
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
index = pc.Index(index_name)
time.sleep(1)
index.describe_index_stats()

import pandas as pd
pd.set_option("display.max_colwidth", None)

import datasets
from datasets import load_dataset
ds = datasets.load_dataset("text", data_files="summary1.rtf", split="train")

from tqdm.notebook import tqdm
from langchain.docstore.document import Document
RAW_KNOWLEDGE_BASE = [
    Document(page_content=doc["text"])
    for doc in tqdm(ds)
    ]

MARKDOWN_SEPARATORS = [
    "\n#{1,6}",
    "'''\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n__+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

from sentence_transformers import SentenceTransformer
print(f"Model's maximum sequence length: {SentenceTransformer('consciousAI/cai-lunaris-text-embeddings').max_seq_length}")
model=SentenceTransformer('consciousAI/cai-lunaris-text-embeddings')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("consciousAI/cai-lunaris-text-embeddings")

from typing import Optional, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
EMBEDDING_MODEL_NAME = "consciousAI/cai-lunaris-text-embeddings"
def split_documents(chunk_size: int, knowledge_base: List[Document],
                    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique

docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE, tokenizer_name=EMBEDDING_MODEL_NAME)

from striprtf.striprtf import rtf_to_text
def clean_rtf(rtf_text: str) -> str:
    return rtf_to_text(rtf_text)

from langchain.prompts import PromptTemplate
def generate_response(question, retrieved_docs, max_tokens=4096):
    print("Question:")
    print(question)
    print("Context:")
    print(retrieved_docs)
    template = f"""You are a knowledgeable assistant. Use the provided documents to answer the question accurately.

    Question: {question}

    Retrieved Documents:{retrieved_docs}

    Answer:
    """
    print("Response:\n")
    prompt_template = PromptTemplate(template=template, input_variables=["question", "retrieved_docs"])
    prompt = prompt_template.format(question=question,retrieved_docs=retrieved_docs)
    response= model_llama(prompt)
    print(response)
    return response

import requests
from requests.exceptions import Timeout
class TimeoutException(Exception):
    pass

@app.route('/ans', methods=['POST'])
def ans():
    try:
        data = request.json
        question = data['question']
        query_embedding = model.encode(question)
        query_embedding = query_embedding.tolist()
        results = index.query(vector=query_embedding, top_k=5)

        context = ""
        for result in results["matches"]:
            doc_index = int(result["id"].split("_")[1])
            context += (clean_rtf(docs_processed[doc_index].page_content))
        response = generate_response(question, context)
        return jsonify({'response': response})

    except TimeoutException:
        return jsonify({'error': 'Request timed out'})

from flask import Flask, request, jsonify, Response
import time

app = Flask(__name__)

@app.route('/ans', methods=['POST'])
def ans():
    def generate():
        try:
            # Notify the client that the process has started
            yield "Processing request...\n"

            # Retrieve data from the POST request
            data = request.json
            question = data['question']

            # Simulate long-running processing (replace this with actual logic)
            time.sleep(5)  # Example delay to mimic model processing
            query_embedding = model.encode(question)
            query_embedding = query_embedding.tolist()
            results = index.query(vector=query_embedding, top_k=5)

            # Build the context for generating the response
            context = ""
            for result in results["matches"]:
                doc_index = int(result["id"].split("_")[1])
                context += clean_rtf(docs_processed[doc_index].page_content)

            # Generate the final response
            response = generate_response(question, context)

            # Yield the final response in JSON format
            yield jsonify({'response': response}).data.decode('utf-8') + "\n"

        except TimeoutException:
            # Notify the client if a timeout occurs
            yield jsonify({'error': 'Request timed out'}).data.decode('utf-8') + "\n"

    # Return a streamed response
    return Response(generate(), content_type='application/json')

from pyngrok import ngrok
if __name__ == '__main__':
    ngrok.set_auth_token(userdata.get('ngrok_auth_token'))
    public_url = ngrok.connect(5000)
    print('Public URL:', public_url)

app.run()