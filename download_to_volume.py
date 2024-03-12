from utils.modal_utils import stub, volume, images
import os


@stub.function(volumes={"/data": volume}, image=images['download_model'])
def download_model(model_file_name, download_url):
    
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