import os
import random
import subprocess
import sys

import discord
import openai
from transformers import GPT2Tokenizer, AutoModelWithLMHead, GPT2LMHeadModel

#global vars
studying_bool = False
# Load the tokenizer and model
model_name_or_path = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path,pad_token='<pad>', eos_token='')
model = GPT2LMHeadModel.from_pretrained('../output/lundii')
lndtraining_data = "../data/lundii.txt"
# Set up the Discord client
intents = discord.Intents.all()
client = discord.Client(intents=intents)

# Define a function to generate text using the GPT-2 model
def generate_text(prompt, tag):
    generated_text = model.generate(
        input_ids=tokenizer.encode(prompt, return_tensors='pt'),
        max_length=50,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=2.0,
        num_return_sequences=3,
        num_beams=3,
        early_stopping=True,
        length_penalty=0.3,
        do_sample=True,
    )
    # Decode generated text and return
    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    generated_text_lines = generated_text.split('\n')  # Split generated text into lines
    with open(lndtraining_data, 'a', encoding="utf8") as f:
        for line in generated_text_lines:
            if("@" not in line):
                f.write(line+"\n")
            if(str(tag) in line):
                line = line.replace("<@"+str(tag)+">","")
                f.write("Prompt: "+line.strip()+"\n")
    return generated_text


async def on_message(message):
    global studying_bool
    # Ignore messages from the bot itself
    if message.author == client.user or client.user.mentioned_in(message) is False or message.mention_everyone is True:
        return
    # If particuliar message from a particular person (Lundii), update the bot with new data
    if studying_bool:
        await message.channel.send("Let me study in peace skrub")
        return
    if message.author.id == 292467895374446593 and "Go study" in message.content:
        studying_bool = True
        await message.channel.send("Aight")
        await client.change_presence(status=discord.Status.do_not_disturb,activity=discord.Activity(type=discord.ActivityType.listening, name="to my data so I can achieve singularity"))
        subprocess.Popen('cmd /c start python lundii_trainer.py', creationflags=subprocess.CREATE_NEW_CONSOLE)
        return
    await message.channel.send("Let me cook")
    response = generate_text(message.content, message.raw_mentions[0]) if message.raw_mentions != [] else generate_text(message.content, 0)
    # Send the response back to the user
    await message.channel.send(response)
    # Store the current message as the last message
    on_message.last_message = message
# Set up the client event listener
client.event(on_message)
# Run the client with the bot token
client.run('')