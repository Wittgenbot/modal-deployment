from utils.modal_utils import stub, volume, images


@stub.function()
def query_model_ctransformers(repo_id, model_file_name, question):

    from ctransformers import AutoModelForCausalLM

    model_file_path = repo_id.replace('/', '_') + '/' + model_file_name

    print(f"Loading model {model_file_name} into GPU...")

    model = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=f'/data/models/{model_file_path}',
                                                 gpu_layers=64,
                                                 context_length=6200,
                                                 local_files_only=True
                                                 )
    
    print("Model successfully loaded.")

    print("Querying model...")

    response = model(question)

    print("Response generated.")

    return response

# change image name!!
@stub.function(volumes={"/data": volume}, image=images['query_model'], gpu="t4")
def query_mistral_7b_instruct_v0p2(question):

    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file_name = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"

    response = stub.registered_functions['query_model_ctransformers'].local(repo_id, model_file_name, question)

    return response


@stub.function(volumes={"/data": volume}, image=images['query_wittgenbot_ft'], gpu="t4")
def query_wittgenbot_ft(question):

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    wittgenbot_model_id = 'descartesevildemon/Wittgenbot-7B'
    wittgenbot_file_path = '/data/models/' + wittgenbot_model_id.replace('/', '_')
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_use_double_quant=True)


    print(f"Loading model {wittgenbot_model_id} into GPU...")

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=wittgenbot_file_path,
                                                 quantization_config=bnb_config,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map='auto',
                                                 local_files_only=True)

    print("Model successfully loaded.")

    print(f"Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=wittgenbot_file_path)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False

    print("Tokenizer successfully loaded.")

    print("Querying model...")

    prompt_template = """
    <s>
    [INST]
    You are a philosophy professor specialized in Ludwig Wittgenstein. Your task is to answer the following question about Ludwig Wittgenstein in a conversational, clear and coherent tone:
    {question}
    [/INST]
    """
    prompt = prompt_template.format(question=question)

    encodeds = tokenizer(prompt,
                        return_tensors="pt",
                        add_special_tokens=True)

    model_inputs = encodeds.to('cuda:0')

    generated_ids = model.generate(**model_inputs,
                                    max_new_tokens=1000,
                                    do_sample=True,
                                    top_p=0.9,
                                    temperature=0.7,
                                    pad_token_id=tokenizer.eos_token_id)

    decoded = tokenizer.batch_decode(generated_ids)
    
    response = decoded[0][len(prompt):-4].strip()

    print("Response generated.")

    return response