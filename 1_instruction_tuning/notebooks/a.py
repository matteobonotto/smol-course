import os, sys

sys.path.append(os.getcwd())



from huggingface_hub import login
# login()


# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device = "cuda:1"


# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    device_map=device,
)#.to(device)

print(model.device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to(device)
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-MyDataset"
finetune_tags = ["smol-course", "module_1"]




# Let's test the base model before training
prompt = "Write a haiku about programming"

# Format with template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print("Before training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



# Load a sample dataset
from datasets import load_dataset

# TODO: define your dataset and config using the path and name parameters
ds = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
ds = load_dataset(
    "bigcode/the-stack-smol", 
    split="train[0:100]",
)


# Configure the SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_steps=200,  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=2,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=10,  # Frequency of logging training metrics
    save_steps=50,  # Frequency of saving model checkpoints
    eval_strategy="no",  # Evaluate the model at regular intervals
    # eval_steps=0,  # Frequency of evaluation
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
    hub_model_id=finetune_name,  # Set a unique name for your model
)


# def format_cols(sample):
#     return {"text":sample["question"], "label":sample["answer"]}

# ds = ds.map(format_cols)


# dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
# ds = load_dataset("bigcode/the-stack-smol", split="train", data_dir="data/python")
# ds = load_dataset("davanstrien/haiku_dpo", split="train")
ds = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations", split="train")
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

def format_ds_text_label(sample):
    return {"text":sample["question"], "label":sample["chosen"]}
ds = ds.map(formatting_prompts_func)

# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['instruction'])):
#         text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
#         output_texts.append(text)
#     return output_texts

# formatting_prompts_func(next(iter(ds)))

# response_template = " ### Answer:"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# trainer = SFTTrainer(
#     model,
#     train_dataset=dataset,
#     args=SFTConfig(output_dir="/tmp"),
#     formatting_func=formatting_prompts_func,
#     data_collator=collator,
# )

# Initialize the SFTTrainer
sft_config.device = device
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds,
    tokenizer=tokenizer,
    eval_dataset=ds,
    # dataset_text_field="content",  # Specify the correct text field name
)


# TODO: 🦁 🐕 align the SFTTrainer params with your chosen dataset. For example, if you are using the `bigcode/the-stack-smol` dataset, you will need to choose the `content` column`

# Train the model
trainer.train()

# Save the model
trainer.save_model(f"./{finetune_name}")

# x = [x if x != -100 else 0 for x in inputs['input_ids'][0,...].cpu().tolist()]
# print(self.tokenizer.decode(x))
# print("\n\n")

# x = [x if x != -100 else 0 for x in inputs['labels'][0,...].cpu().tolist()]
# print(self.tokenizer.decode(x))


# inputs['input_ids'][0,...].cpu().tolist()
# inputs['labels'][0,...].cpu().tolist()


# Test the fine-tuned model on the same prompt

# Let's test the base model before training
prompt = "Write a haiku about programming"

# Format with template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

# TODO: use the fine-tuned to model generate a response, just like with the base example.
outputs = model.generate(**inputs, max_new_tokens=100)
print("After training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))













