from modal import Stub, Volume, Image


stub = Stub("wittgenbot")


volume = Volume.from_name("fyp-volume")


images = {
    "download_from_hf": (
        Image.debian_slim(python_version="3.10")
        .pip_install("huggingface_hub")
    ),
    "download_embedding_model": (
        Image.debian_slim(python_version="3.10")
        .pip_install("transformers", "torch")
    ),
    "query_model": (
        Image.debian_slim(python_version="3.10")
        .pip_install("ctransformers", "torch")
    ),
    "query_wittgenbot_ft": (
        Image.debian_slim(python_version="3.10")
        .pip_install("transformers", "torch", "accelerate", "bitsandbytes")
    ),
    "semantic_search": (
        Image.debian_slim(python_version="3.10")
        .pip_install("langchain", "sentence-transformers", "faiss-cpu")
    )
}