from transformers import MambaConfig, MambaModel, BertTokenizer
from torch.nn import LSTMCell
config = MambaConfig()
model = MambaModel(config)
tokenizer = BertTokenizer.from_pretrained("./Documents/competition/BC-AVSMod/pretrained_llms/bert-base-uncased", do_lower_case=True)
inputs = tokenizer("This image is apple", add_special_tokens=True, max_length=50, padding="max_length", truncation=True, return_tensors="pt")

out = model(inputs["input_ids"])
print(out.shape)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType