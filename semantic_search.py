from utils.modal_utils import stub, volume, images


@stub.function(volumes={"/data": volume}, image=images['semantic_search'])
def semantic_search(question, faiss_index_type, num_matched_excerpts):

    from langchain_community.vectorstores import FAISS

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