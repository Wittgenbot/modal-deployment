import modal
from dotenv import load_dotenv

load_dotenv()

# Downloading Wittgenbot-7B to volume
# download_hf_repo = modal.Function.lookup("wittgenbot", "download_hf_repo")
# wittgenbot_repo_id = "descartesevildemon/Wittgenbot-7B"
# download_hf_repo.remote(wittgenbot_repo_id)

# Downloading Mistral-7B-Instruct-v0.2-GGUF to volume
# download_hf_file = modal.Function.lookup("wittgenbot", "download_hf_file")
# mistral_repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# mistral_file_name = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
# download_hf_file.remote(mistral_repo_id, mistral_file_name)

# Downloading embedding model to volume
# download_embedding_model = modal.Function.lookup("wittgenbot", "download_embedding_model")
# embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# download_embedding_model.remote(embedding_model_name)

# Query Wittgenbot
query_wittgenbot_ft = modal.Function.lookup("wittgenbot", "query_wittgenbot_ft")
question = 'According to Ludwig Wittgenstein\'s philosophy, is the existence of a private language possible?'
response = query_wittgenbot_ft.remote(question)
print(response)

# Query Mistral-7B-Instruct-v0.2-GGUF
# query_mistral_7b_instruct_v0p2 = modal.Function.lookup("wittgenbot", "query_mistral_7b_instruct_v0p2")
# question = 'Why is the sky blue?'
# response = query_mistral_7b_instruct_v0p2.remote(question)
# print(response)

# Run semantic search
# semantic_search = modal.Function.lookup("wittgenbot", "semantic_search")
# question = "What is Wittgenstein's private language argument?"
# index_type = {
#     'chunk_size': 512,
#     'chunk_overlap': 50
# }
# num_matched_excerpts = 2
# documents = semantic_search.remote(question, index_type, num_matched_excerpts)
# print(documents)