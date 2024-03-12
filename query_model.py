from utils.modal_utils import stub, volume, images


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


@stub.function(volumes={"/data": volume}, image=images['query_model'], gpu="t4")
def query_mistral_7b_instruct_v0p2(prompt):

    model_file_name = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"

    response = stub.registered_functions['query_model'].local(model_file_name, prompt)

    return response