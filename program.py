import os
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, AutoModelWithLMHead
import openai
import torch
from datasets import load_dataset
load_dataset("wikipedia", "20220301.en")