import os
import subprocess
import sys
import validators
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, AutoModelWithLMHead
source = "discordPete.py"
file = open('../data/dms.txt', 'r', encoding="utf8")
Lines = file.readlines()
text = ""
for i, line in enumerate(Lines):
    if (line != ""):
        if (line.count("Trolltusk#")):
            if("https" not in Lines[i + 1] and "cdn" not in Lines[i + 1]):
               text += Lines[i + 1]

y = open("../data/Oxford English Dictionary.txt", encoding="utf8")
f = open("../data/pete.txt", "w", encoding="utf8")
f.write(text+"\n"* 2)
#adding more weight to the Oxford English Dictionary
f.write(y.read())
f.close()
print(text)
# Load the tokenizer and model
model_name_or_path = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, pad_token='<pad>', eos_token='')
model = GPT2LMHeadModel.from_pretrained(model_name_or_path, return_dict=True)
# Load the training data
train_dataset = TextDataset(tokenizer=tokenizer, file_path="../data/pete.txt", block_size=512)
# Set up the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the training arguments
training_args = TrainingArguments(
        output_dir='../output',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=6,
        save_steps=150,
        logging_steps=150,
        save_total_limit=6,
        weight_decay=0.1,
        warmup_steps=130,
        warmup_ratio=1
    )
# Set up the Trainer and start training
trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
trainer.train()
os.system('python ' + source)
sys.exit()
#os.execl(sys.executable, 'python', source)