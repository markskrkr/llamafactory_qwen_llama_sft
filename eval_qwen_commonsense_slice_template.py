# Evaluate Qwen2.5-7B SliceFine checkpoint on Commonsense tasks (BoolQ).
# Adapted from dora_evaluate.py with SliceFine checkpoint loading.
# Modification: Applied Qwen2.5 Chat Template to fix instruction following issues.

import copy
import json
import os
import re
import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Default paths
DEFAULT_BASE_MODEL_PATH = "/mnt/workspace/models/open_source/Qwen2.5-1.5B"
DEFAULT_CHECKPOINT_PATH = "outputs/qwen2_5_1.5b_commonsense_slice_n64_m20"

# Dataset paths mapping
DATASET_PATHS = {
    "boolq": "glue_local_data/commonsence/boolq/validation-00000-of-00001.parquet",
    "piqa": "glue_local_data/commonsence/piqa/valid.jsonl",
    "social_i_qa": "glue_local_data/commonsence/siqa/socialIQa_v1.4_dev.jsonl",
    "hellaswag": "glue_local_data/commonsence/hellaswag/validation-00000-of-00001.parquet",
    "winogrande": "glue_local_data/commonsence/winogrande/validation-00000-of-00001.parquet",
    "ARC-Challenge": "glue_local_data/commonsence/ARC-Challenge/test-00000-of-00001.parquet",
    "ARC-Easy": "glue_local_data/commonsence/ARC-Easy/test-00000-of-00001.parquet",
    "openbookqa": "glue_local_data/commonsence/OBQA/test-00000-of-00001.parquet",
}

def extract_answer(dataset_name: str, sentence: str) -> str:
    """Extract answer from dora_evaluate.py."""
    sentence_ = sentence.strip().lower()
    if dataset_name == 'boolq':
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset_name == 'piqa':
        pred_answers = re.findall(r'solution\s*[12]', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0].replace(" ", "")
    elif dataset_name in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        pred_answers = re.findall(r'answer\s*[1-5]', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0].replace(" ", "")
    elif dataset_name == 'hellaswag':
        pred_answers = re.findall(r'ending\s*[1-4]', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0].replace(" ", "")
    elif dataset_name == 'winogrande':
        pred_answers = re.findall(r'option\s*[12]', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0].replace(" ", "")
    return ""


def convert_to_dora_format(dataset_name, dataset, labels=None):
    """Convert dataset to dora_evaluate.py format with instruction and answer fields."""
    data_list = []

    if dataset_name == 'boolq':
        for example in dataset:
            instruction = f"{example['passage']}\nQuestion: {example['question']}\nAnswer: true or false?"
            label = "true" if example["answer"] else "false"
            data_list.append({"instruction": instruction, "answer": label})

    elif dataset_name == 'piqa':
        for i, example in enumerate(dataset):
            instruction = f"{example['goal']}\nSolution1: {example['sol1']}\nSolution2: {example['sol2']}\nThe better solution is solution1 or solution2?"
            label = "solution1" if labels[i] == 0 else "solution2"
            data_list.append({"instruction": instruction, "answer": label})

    elif dataset_name == 'social_i_qa':
        for example in dataset:
            instruction = f"{example['context']}\nQuestion: {example['question']}\nAnswer1: {example['answerA']}\nAnswer2: {example['answerB']}\nAnswer3: {example['answerC']}\nThe most suitable answer is answer1, answer2, or answer3?"
            label_map = {"A": "answer1", "B": "answer2", "C": "answer3"}
            label = label_map[example["correct"]]
            data_list.append({"instruction": instruction, "answer": label})

    elif dataset_name == 'hellaswag':
        for example in dataset:
            endings = eval(example['endings']) if isinstance(example['endings'], str) else example['endings']
            instruction = f"{example['ctx']}\nEnding1: {endings[0]}\nEnding2: {endings[1]}\nEnding3: {endings[2]}\nEnding4: {endings[3]}\nThe most suitable ending is ending1, ending2, ending3, or ending4?"
            label = f"ending{int(example['label']) + 1}"
            data_list.append({"instruction": instruction, "answer": label})

    elif dataset_name == 'winogrande':
        for example in dataset:
            instruction = f"{example['sentence']}\nOption1: {example['option1']}\nOption2: {example['option2']}\nThe suitable option is option1 or option2?"
            label = f"option{example['answer']}"
            data_list.append({"instruction": instruction, "answer": label})

    elif dataset_name in ['ARC-Challenge', 'ARC-Easy']:
        for example in dataset:
            choices_text = eval(example['choices']['text']) if isinstance(example['choices']['text'], str) else example['choices']['text']
            choices_label = eval(example['choices']['label']) if isinstance(example['choices']['label'], str) else example['choices']['label']
            answers_str = "\n".join([f"Answer{i+1}: {t}" for i, t in enumerate(choices_text)])
            options_str = ", ".join([f"answer{i+1}" for i in range(len(choices_text))])
            instruction = f"{example['question']}\n{answers_str}\nThe correct answer is {options_str}?"
            label_idx = choices_label.index(example['answerKey'])
            label = f"answer{label_idx + 1}"
            data_list.append({"instruction": instruction, "answer": label})

    elif dataset_name == 'openbookqa':
        for example in dataset:
            choices_text = eval(example['choices']['text']) if isinstance(example['choices']['text'], str) else example['choices']['text']
            choices_label = eval(example['choices']['label']) if isinstance(example['choices']['label'], str) else example['choices']['label']
            answers_str = "\n".join([f"Answer{i+1}: {t}" for i, t in enumerate(choices_text)])
            options_str = ", ".join([f"answer{i+1}" for i in range(len(choices_text))])
            instruction = f"{example['question_stem']}\n{answers_str}\nThe correct answer is {options_str}?"
            label_idx = choices_label.index(example['answerKey'])
            label = f"answer{label_idx + 1}"
            data_list.append({"instruction": instruction, "answer": label})

    return data_list


def load_dataset_by_name(dataset_name, dataset_path):
    """Load dataset based on name and path, return (dataset, labels)."""
    labels = None

    if dataset_name == 'piqa':
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        labels_path = dataset_path.replace('.jsonl', '-labels.lst').replace('valid.', 'valid-labels.')
        if 'valid.jsonl' in dataset_path:
            labels_path = dataset_path.replace('valid.jsonl', 'valid-labels.lst')
        with open(labels_path, 'r') as f:
            labels = [int(line.strip()) for line in f]
        return data, labels

    elif dataset_name == 'social_i_qa':
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data, None

    else:
        ds = load_dataset("parquet", data_files={"validation": dataset_path})["validation"]
        return ds, None


def create_batch(dataset, batch_size):
    """From dora_evaluate.py."""
    batches = []
    num_batch = len(dataset) // batch_size if len(dataset) % batch_size == 0 else len(dataset) // batch_size + 1
    for i in range(num_batch):
        batch = dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SliceFine checkpoint on Commonsense tasks")
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        default="boolq")
    parser.add_argument('--base_model', type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help="Path to SliceFine checkpoint (set to empty string for base model only)")
    parser.add_argument('--dataset_path', type=str, default=None,
                        help="Path to dataset (auto-detected if not provided)")
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = DATASET_PATHS[args.dataset]
    return args


def load_model(args):
    """Load SliceFine checkpoint (full model, not PEFT)."""
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 0

    model_path = args.checkpoint_path if args.checkpoint_path else args.base_model
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    
    # Setup evaluate function (closure over tokenizer/model)
    tokenizer, model = load_model(args)
    
    def evaluate(instructions, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=64):
        """
        Modified generation logic to support Qwen2.5 Chat Template.
        Using apply_chat_template avoids 'repetition/completion mode' on SFT models.
        """
        prompts = []
        for instruction in instructions:
            # Construct ChatML format messages
            # Note: Qwen2.5 works best when we explicitly set role: system
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ]
            # apply_chat_template handles the <|im_start|>... logic automatically
            # add_generation_prompt=True ensures it ends with <|im_start|>assistant to trigger generation
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(text)

        # Tokenize with left padding for batched generation
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # repetition_penalty=1.1, # Optional: enable if model still repeats slightly
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False, # Set to False to save memory if not needed
                max_new_tokens=max_new_tokens,
            )
        
        # SLICE OUTPUT: Only decode the *newly generated* tokens.
        # This prevents the need to split by "### Response:" which doesn't exist in ChatML output.
        # input_ids.shape[1] is the length of the prompt.
        generated_ids = generation_output.sequences[:, input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Log for debugging
        print(f"Debug - Raw output: {outputs[0]}")
        
        # Clean up whitespace
        outputs = [o.strip() for o in outputs]
        return outputs

    # Setup output
    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path.rstrip('/')
        parent_dir = os.path.basename(os.path.dirname(ckpt_path))
        ckpt_name = os.path.basename(ckpt_path)
        if ckpt_name.startswith('checkpoint-'):
            model_name = f"{parent_dir}_{ckpt_name}"
        else:
            model_name = ckpt_name
    else:
        model_name = os.path.basename(args.base_model.rstrip('/'))
    save_file = f'experiment_template/{model_name}-{args.dataset}.json'
    create_dir('experiment_template/')

    # Load and convert dataset
    print(f"Loading dataset from: {args.dataset_path}")
    raw_dataset, labels = load_dataset_by_name(args.dataset, args.dataset_path)
    dataset = convert_to_dora_format(args.dataset, raw_dataset, labels)
    print(f"Total samples: {len(dataset)}")
    
    batches = create_batch(dataset, args.batch_size)

    # Evaluation loop
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]
        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args.dataset, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            
            # Print less verbose logs during run
            if idx % 10 == 0: 
                print(f"\nInstr: {data['instruction'][:80]}...")
                print(f"Pred: {predict} | Label: {label}")

        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current:.4f}', end='')
        
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    
    pbar.close()
    print('\n')
    print('test finished')
    print(f"Final accuracy: {correct}/{current} = {correct/current:.4f}")


if __name__ == "__main__":
    main()