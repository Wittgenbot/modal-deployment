from utils import stub, volume, images
import os


@stub.function(volumes={"/data": volume}, image=images['download_from_hf'])
def download_hf_repo(repo_id):
    
    from huggingface_hub import snapshot_download

    save_directory = '/data/models/'
    file_path = save_directory + repo_id.replace('/','_')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory {save_directory}")

    print(f'Downloading repository {repo_id}')   
    snapshot_download(repo_id=repo_id,
                      local_dir_use_symlinks=False,
                      local_dir=file_path
    )

    print(f'Committing changes to volume.')
    volume.commit()
    print('Changes committed.')


@stub.function(volumes={"/data": volume}, image=images['download_from_hf'])
def download_hf_file(repo_id, file_name):
    
    from huggingface_hub import hf_hub_download

    save_directory = '/data/models/'
    file_path = save_directory + repo_id.replace('/','_')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory {save_directory}")

    print(f'Downloading file {file_name} from {repo_id}')   
    hf_hub_download(repo_id=repo_id,
                    filename=file_name,
                    local_dir_use_symlinks=False,
                    local_dir=file_path
    )

    print(f'Committing changes to volume.')
    volume.commit()
    print('Changes committed.')


@stub.function(volumes={"/data": volume}, image=images['download_embedding_model'])
def download_embedding_model(embedding_model_name):

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
    volume.commit()
    print('Changes committed.')