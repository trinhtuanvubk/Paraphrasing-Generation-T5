from datasets import load_dataset
import torch, nltk
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
nltk.download("punkt", quiet=True)

from .preprocess import preprocess_function

def data_loader(args, tokenizer):
    train_dataset = load_dataset(args.dataset_name, split="train").shuffle(seed=42).select(range(50000))
    val_dataset = load_dataset(args.dataset_name, split="train").shuffle(seed=42).select(range(50000,55000))

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizered_train = train_dataset.map(lambda batch: preprocess_function(args, batch, tokenizer), batched=True)
    tokenizered_val = val_dataset.map(lambda batch: preprocess_function(args, batch, tokenizer), batched=True)

    return tokenizered_train, tokenizered_val




    