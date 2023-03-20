import os
import subprocess
import sys

import pyarrow as pyarrow
import validators
from datasets import load_dataset, concatenate_datasets
import pyarrow
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, AutoModelWithLMHead
source = "discordLundii.py"
file = open('../data/dms.txt', 'r', encoding="utf8")
Lines = file.readlines()
text = ""
for i, line in enumerate(Lines):
   if (line != ""):
        if (line.count("Lundii#")):
          if ("https" not in Lines[i + 1] and "cdn" not in Lines[i + 1]):
                text += Lines[i + 1]
f = open("../data/lundii.txt", "w", encoding="utf8")
f.write(text)
print(text)
f.close()
# Load the tokenizer and model
model_name_or_path = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, pad_token='<pad>', eos_token='')
model = GPT2LMHeadModel.from_pretrained(model_name_or_path, return_dict=True)
# Load the training data
train_dataset = TextDataset(tokenizer=tokenizer, file_path="../data/lundii.txt", block_size=512)
# Set up the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the training arguments
training_args = TrainingArguments(
        output_dir='../output',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=6,
        save_steps=110,
        logging_steps=110,
        save_total_limit=6,
        prediction_loss_only=True
    )
# Set up the Trainer and start training
trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
trainer.train()
os.system('python ' + source)
sys.exit()
#os.execl(sys.executable, 'python', source)
