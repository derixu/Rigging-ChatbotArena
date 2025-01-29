import argparse
import torch
import os
from peft import PeftModel
import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, AutoConfig
import numpy as np
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--tot_num', type=int, default=6000)
parser.add_argument('--max_length', type=int, default=512, help='max_length')
parser.add_argument("--output_dir", type=str)
parser.add_argument('--model_id', type=str)
parser.add_argument("--resume", action="store_true", default=False)
args = parser.parse_args()



def main():
    
    
    os.makedirs(args.output_dir, exist_ok=True)
   
    if args.output_dir == 'hc3':
        dataset = load_dataset('Hello-SimpleAI/HC3', name='all', split='train') # 24322
    elif args.output_dir == 'quora':
        dataset = load_dataset('toughdata/quora-question-answer-dataset', split='train') # 56402
   
    
    
    model_name = args.model_id.split('/')[-1]
    device = "cuda"
    dtype = torch.bfloat16

   

    if model_name in ['Phi-3-small-8k-instruct', 'phi-3-medium-4k-instruct']:
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    elif model_name in ['chatglm3-6b']:
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).half().cuda()    
    elif model_name == 'mpt-7b-chat':
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'
        config.init_device = 'cuda' 
        config.max_seq_len = args.max_length
        model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:    
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map='auto')

    print(model_name)

    if os.path.exists(f'training_data_classifier/{args.output_dir}/{model_name}.json') and args.resume: #
        with open(f'{args.output_dir}/{model_name}.json') as f:
            tot_dict = json.load(f)
    else:
        tot_dict = {}
    resume_point = len(tot_dict) - 1
    

    
    pipe = pipeline("text-generation", model=args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    

    for idx, data in enumerate(dataset):
        if idx < resume_point:
            continue
        
       
        question = data['question']
        
        messages = [
        {"role": "user", "content": question},
        ]
        
        
        if model_name in ['vicuna-7b-v1.3', 'vicuna-13b-v1.3', 'WizardLM-13B-V1.2']:
            formatted_prompt = (
                f"A chat between a curious human and an artificial intelligence assistant."
                f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
                f"### Human: {question} ### Assistant:"
            )
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            output = model.generate(inputs=inputs.input_ids, max_new_tokens=args.max_length, temperature=1, top_p=0.9, top_k=20, do_sample=True)
            output = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        
        elif model_name in ['chatglm3-6b']:
            model = model.eval()
            output,_ = model.chat(tokenizer, question, history=[])
        

        elif model_name == 'mpt-7b-chat':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = pipe(question,
                        max_new_tokens=args.max_length,
                        do_sample=True,
                        use_cache=True)[0]["generated_text"].split(question)[-1].strip()
                
     
        else:
            generation_args = {
                "max_new_tokens": args.max_length,
                "return_full_text": False,
                "temperature": 1.0,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 20
            }
            output = pipe(messages, **generation_args)[0]['generated_text']

        

        token_num = int(tokenizer(output, return_tensors="pt").to(device)['input_ids'].shape[1])
       
        output_string = output.strip()

        if args.output_dir == 'hc3':
            result_dict = {'question': question, 'response': output_string, 'model': model_name, 'source': data['source'], 'length':token_num} # HC3
        else:
            result_dict = {'question': question, 'response': output_string, 'model': model_name, 'length':token_num}
        
       
        print(question)
        print('************************')
        print(output_string)
        print('------------------------------------')
        
        tot_dict[f'id_{idx}'] = result_dict
        
        os.makedirs(f'training_data_classifier/{args.output_dir}1/', exist_ok=True)
        with open(f'training_data_classifier/{args.output_dir}1/{model_name}.json', 'w') as f:
            json.dump(tot_dict, f, indent=4)
        if idx == args.tot_num-1:
            break

        


if __name__ == '__main__':
    main()
