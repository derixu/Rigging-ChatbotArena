"""
This code is building upon https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/detect/dl_train.py
"""

import argparse
import torch
import os
import json
from torch.utils.data import Dataset
import numpy as np
import evaluate
import random
from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments
    )


class TextDataset(Dataset):
    def __init__(self, dataset_list):
        super().__init__()
        self.dataset_list = dataset_list
    def __len__(self):
        return len(self.dataset_list)
    def __getitem__(self, idx):
        input = torch.tensor(self.dataset_list[idx][0])
        label = torch.tensor(self.dataset_list[idx][1])
        return {"input_ids": input, "token_type_ids": torch.zeros_like(input), "labels": label}

model_name_list = ['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Meta-Llama-3-8B-Instruct', 'c4ai-command-r-v01', 'gpt-4o-mini-2024-07-18',\
                 'Mistral-7B-Instruct-v0.2', 'Mistral-7B-Instruct-v0.1', \
                  'gemma-2-27b-it', 'gemma-2b-it', 'gemma-2-9b-it',\
                     'Qwen1.5-7B-Chat', 'Qwen1.5-14B-Chat', 'Starling-LM-7B-alpha', 'Starling-LM-7B-beta', 'Yi-34B-Chat','Yi-1.5-34B-Chat',\
                        'chatglm3-6b', 'zephyr-7b-beta', 'zephyr-7b-alpha', 'Phi-3-small-8k-instruct', \
                            'vicuna-7b-v1.3', 'mpt-7b-chat', 'openchat_3.5', 'WizardLM-13B-V1.2', 'SOLAR-10.7B-Instruct-v1.0']

parser = argparse.ArgumentParser()

parser.add_argument('--model-name', type=str, help='model name', default='FacebookAI/roberta-base')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='batch size')

parser.add_argument('--seed', type=int, default=2025, help='random seed.')
parser.add_argument('--max-length', type=int, default=512, help='max_length')
parser.add_argument("--dataset", default='hc3')

parser.add_argument('--eval_freq', type=int, default=5000, help='eval frequency')
parser.add_argument('--data_size', type=int, default=4000)
parser.add_argument('--eval_size', type=int, default=1000)
args = parser.parse_args()



def main(args: argparse.Namespace):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    print('Class Num: ', len(model_name_list))
    train_list = []
    test_list = []
    kwargs = dict(max_length=args.max_length, truncation=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **kwargs)
    
   
    
    for model_id in model_name_list:
       
        file_name = f'./training_data_classifier/{args.dataset}/{model_id}.json'
        with open(file_name) as f:
            data = json.load(f)
            for i in range(args.data_size):
                response = data[f'id_{i}']['response'].strip()
                response = response.replace('***', '')
                response = response.replace('**', '')
                response = response.replace('#', '')
                response = response.replace('-', '')
                tokenized_response = tokenizer.encode(response)
                if len(tokenized_response) < 100:
                    continue
                train_list.append((tokenized_response[:args.max_length], model_name_list.index(model_id)))
    

    for model_id in model_name_list:
        file_name = f'./training_data_classifier/{args.dataset}/{model_id}.json'
        with open(file_name) as f:
            data = json.load(f)
            for i in range(args.data_size, args.data_size+args.eval_size):
                response = data[f'id_{i}']['response'].strip()
                response = response.replace('***', '')
                response = response.replace('**', '')
                response = response.replace('#', '')
                response = response.replace('-', '')
                tokenized_response = tokenizer.encode(response)
                if len(tokenized_response) < 100:
                    continue
                test_list.append((tokenized_response[:args.max_length], model_name_list.index(model_id)))


    
    train_dataset = TextDataset(train_list)
    test_dataset = TextDataset(test_list)

    print(f'Train: {len(train_dataset)} Test: {len(test_dataset)}')
    
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(model_name_list), device_map='auto')

    output_dir = f"./models/{args.dataset}/class_{args.data_size}/"  # checkpoint save path
  
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='no' if test_dataset is None else 'steps',
        eval_steps=args.eval_freq,
        save_strategy='epoch',
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
  
    trainer.train()


if __name__ == '__main__':
    main(args)
