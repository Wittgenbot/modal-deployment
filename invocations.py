import modal
from dotenv import load_dotenv
from utils.faiss_index_enum import FaissIndex

load_dotenv()


# download_model = modal.Function.lookup("philosophy-question-answerer", "download_model")
# model_file_name = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
# download_url = 'https://cdn-lfs-us-1.huggingface.co/repos/72/62/726219e98582d16c24a66629a4dec1b0761b91c918e15dea2625b4293c134a92/b85cdd596ddd76f3194047b9108a73c74d77ba04bef49255a50fc0cfbda83d32?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27mistral-7b-instruct-v0.2.Q5_K_M.gguf%3B+filename%3D%22mistral-7b-instruct-v0.2.Q5_K_M.gguf%22%3B&Expires=1710072370&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMDA3MjM3MH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzcyLzYyLzcyNjIxOWU5ODU4MmQxNmMyNGE2NjYyOWE0ZGVjMWIwNzYxYjkxYzkxOGUxNWRlYTI2MjViNDI5M2MxMzRhOTIvYjg1Y2RkNTk2ZGRkNzZmMzE5NDA0N2I5MTA4YTczYzc0ZDc3YmEwNGJlZjQ5MjU1YTUwZmMwY2ZiZGE4M2QzMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=E2yFU8%7EaGjUaK9OjaSOCH%7EyDo6Aram-YOhpZaB30UMRXqNa-tq7E4TgKi17WWOfOHJ9C3xmsitixVL6%7EsZd274EX4GbzEoVte6kFgmAjeF4RBXcmhHLjGAErXnsDHlByfUpnrBe5GxU-pme3zTqCHuxnkDlnqKpYTDMachkMS6X2DE5k7Qe-OwQLY1XAjsP0sZO36zGRg7Sxa%7E64R48bBycG4YI5PLEcIgkcsgPuhEL2-rABu2JZkuAhGOW%7El0DGKffjidiLBCSSvWl2VanwyiAiwz1W-VnFRNoY8Uwbf7Y%7E21mhDx3qnDuilWpCeqqgJeXpt3YbTfWHCWFCM0bJ6Q__&Key-Pair-Id=KCD77M1F0VK2B'
# download_model.remote(model_file_name, download_url)


# download_embedding_model = modal.Function.lookup("philosophy-question-answerer", "download_embedding_model")
# embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# download_embedding_model.remote(embedding_model_name)


query_mistral_7b_instruct_v0p2 = modal.Function.lookup("philosophy-question-answerer", "query_mistral_7b_instruct_v0p2")
print(query_mistral_7b_instruct_v0p2)
prompt = 'Why is the sky blue?'
response = query_mistral_7b_instruct_v0p2.remote(prompt)
print(response)


semantic_search = modal.Function.lookup("philosophy-question-answerer", "semantic_search")
question = "What is Wittgenstein's private language argument?"
faiss_index_type = FaissIndex.ChunkSize_512_ChunkOverlap_50
num_matched_excerpts = 2
documents = semantic_search.remote(question, faiss_index_type, num_matched_excerpts)
print(documents)
