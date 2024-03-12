from modal import Stub, Volume, Image


stub = Stub("philosophy-question-answerer")


volume = Volume.from_name("fyp-volume")


images = {
    "download_model": (
        Image.debian_slim(python_version="3.10")
        .pip_install("requests")
    ),
    "download_embedding_model": (
        Image.debian_slim(python_version="3.10")
        .pip_install("transformers", "torch")
    ),
    "query_model": (
        Image.debian_slim(python_version="3.10")
        .pip_install("ctransformers", "torch")
    ),
    "semantic_search": (
        Image.debian_slim(python_version="3.10")
        .pip_install("langchain", "sentence-transformers", "faiss-cpu")
    )
}