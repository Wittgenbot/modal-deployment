from modal import Stub, Volume, Image
import os
from enum import Enum

class FaissIndex(Enum):
    ChunkSize_512_ChunkOverlap_50 = {
        'size': 512,
        'overlap': 50,   
    }
    ChunkSize_1500_ChunkOverlap_300 = {
        'size': 1500,
        'overlap': 300,   
    }

stub = Stub("philosophy-question-answerer")

vol = Volume.from_name("fyp-volume")

image_download_model = (
    Image.debian_slim(python_version="3.10")
    .pip_install("requests")
)

image_download_embedding_model = (
    Image.debian_slim(python_version="3.10")
    .pip_install("transformers", "torch")
)

image_query_model = (
    Image.debian_slim(python_version="3.10")
    .pip_install("ctransformers", "torch")
)

image_semantic_search = (
    Image.debian_slim(python_version="3.10")
    .pip_install("langchain", "sentence-transformers", "faiss-cpu", "torch", "torchvision")
)


@stub.function(volumes={"/data": vol}, image=image_download_model)
def download_model_to_volume(model_file_name, download_url):
    
    import requests

    save_directory = '/data/model/'
    file_path = save_directory + model_file_name

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory {save_directory}")

    print('Downloading model...')
    response = requests.get(download_url, allow_redirects=True)

    if response.status_code == 200:

        print('Model successfully downloaded.')

        print('Saving to volume.')

        with open(file_path, 'wb') as file:
            file.write(response.content)

        print(f"Model file saved to {file_path}")

    else:
        print(f"Failed to download the model file. Status code: {response.status_code}")

    print(f'Committing changes to volume.')
    vol.commit()
    print('Changes committed.')


@stub.function(volumes={"/data": vol}, image=image_download_embedding_model)
def download_embedding_model_to_volume(embedding_model_name):

    from transformers import AutoTokenizer, AutoModel

    save_directory = '/data/embedding_model/'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory {save_directory}")

    print("Downloading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)
    print("Embedding model downloaded.")

    print("Saving to volume.")
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Embedding model have been saved to {save_directory}")

    print(f'Committing changes to volume.')
    vol.commit()
    print('Changes committed.')


@stub.function()
def query_model(model_file_name, prompt):

    from ctransformers import AutoModelForCausalLM

    print(f"Loading model {model_file_name} into GPU...")

    model = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=f'/data/model/{model_file_name}',
                                                 gpu_layers=64,
                                                 context_length=6200,
                                                 local_files_only=True
                                                 )
    
    print("Model successfully loaded.")

    print("Querying model...")

    response = model(prompt)

    print("Model response generated.")

    return response


@stub.function(volumes={"/data": vol}, image=image_query_model, gpu="t4")
def query_mistral_7b_instruct_v0p2(prompt):

    model_file_name = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"

    response = stub.registered_functions['query_model'].local(model_file_name, prompt)

    return response


@stub.function(volumes={"/data": vol}, image=image_semantic_search)
def semantic_search(question, faiss_index_type, num_matched_excerpts):

    from langchain_community.vectorstores import FAISS

    vol.reload()

    chunk_size = faiss_index_type.value['size']
    chunk_overlap = faiss_index_type.value['overlap']
    faiss_index_path = f'/data/faiss_indices/ChunkSize_{chunk_size}_ChunkOverlap_{chunk_overlap}'

    embedding_model = stub.registered_functions['load_embedding_model'].local()

    faiss_index = FAISS.load_local(faiss_index_path,
                                   embedding_model,
                                    allow_dangerous_deserialization=True)

    results = faiss_index.similarity_search(query=question, k=num_matched_excerpts)
    documents = [
        {
            'text': result.page_content,
            'source': result.metadata['source']
        } for result in results
    ]

    return documents


@stub.function()
def load_embedding_model():

    from langchain.embeddings.huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
        model_name='/data/embedding_model/',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'device': 'cpu', 'batch_size': 10}
    )

    return embedding_model