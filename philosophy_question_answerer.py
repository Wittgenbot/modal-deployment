from modal import Stub, Volume, Image
import os

stub = Stub("philosophy-question-answerer")

vol = Volume.from_name("fyp-volume")

image_query_model = (
    Image.debian_slim(python_version="3.10")
    .pip_install("ctransformers", "torch")
)

image_download_model = (
    Image.debian_slim(python_version="3.10")
    .pip_install("requests")
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

        print(f'Saving to volume.')

        with open(file_path, 'wb') as file:
            file.write(response.content)

        print(f"Model file saved to {file_path}")

    else:
        print(f"Failed to download the model file. Status code: {response.status_code}")

    print(f'Committing changes to volume.')
    vol.commit()
    print('Changes committed.')


@stub.function(volumes={"/data": vol}, image=image_query_model, gpu="t4")
def query_model(model_file_name, prompt):

    from ctransformers import AutoModelForCausalLM

    print("Loading model into GPU...")

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