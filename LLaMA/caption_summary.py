from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"                                       # This is crucial for reproducibility

# caption_pth = "./MMSD/captions/AVE_data_batch_10299.json"
caption_pth = "./MMSD/captions/AVE_data_batch_14334.json"
file = open(caption_pth, 'r')
caption_data = json.load(file)

target_dir = "./Documents/competition/BC-AVSMod/results/captions"

model_id = "./Documents/competition/BC-AVSMod/pretrained_llms/meta/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,
    device_map="auto",
)

ast = time.time()
for key in caption_data.keys():
    bst = time.time()
    content = f"Please remove all speculation and association from the following description, leaving only the information that directly describes the image content: {caption_data[key]}"
    messages = [{"role": "user", "content": content},]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True)
    
    with open(f'{target_dir}/AVE_{key}.json', 'w') as json_file:
        json.dump({key: output}, json_file, indent=4) 
        print(output)
        print(f"Image id is f{key}; Batch time using is {time.time() - bst} s.")
    
print(f"Finished!; Time using is {time.time() - ast} s.")
