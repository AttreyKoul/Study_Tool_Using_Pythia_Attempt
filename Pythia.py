import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", device_map="auto")

