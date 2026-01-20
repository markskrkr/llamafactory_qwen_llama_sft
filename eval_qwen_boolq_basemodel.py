"""
Evaluate Qwen2.5-7B base model on BoolQ validation set.
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configuration
MODEL_PATH = "/mnt/workspace/models/open_source/Qwen2.5-7B"
DATASET_PATH = "glue_local_data/boolq/data/validation-00000-of-00001.parquet"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 8

SYSTEM_PROMPT = "You are a helpful assistant for answering yes/no questions based on a passage."


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0
    return tokenizer


def build_boolq_prompt(tokenizer, passage: str, question: str, max_length: int):
    """Build prompt-only ids for generation (no answer appended)."""
    user_content = (
        "Read the following passage and answer the question with 'yes' or 'no'.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}"
    )
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_length,
    )
    return prompt_ids


def extract_answer_from_text(text: str) -> int:
    """Extract yes/no from generated text. Returns: 1=yes, 0=no, -1=unknown."""
    t = text.lower().strip()
    has_yes = "yes" in t
    has_no = "no" in t
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    if has_yes and has_no:
        return 1 if t.find("yes") < t.find("no") else 0
    return -1


def predict_boolq_with_generate(
    model, tokenizer, passage: str, question: str, device, max_length: int, gen_kwargs: dict
) -> int:
    """Use model.generate to produce answer, then extract yes/no."""
    prompt_ids = build_boolq_prompt(tokenizer, passage, question, max_length)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    # Decode generated sequence (skip input prompt portion)
    gen_ids = generated[0].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return extract_answer_from_text(gen_text)


def main():
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Loading dataset from: {DATASET_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(MODEL_PATH)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    # Load dataset
    print("Loading BoolQ validation dataset...")
    dataset = load_dataset("parquet", data_files={"validation": DATASET_PATH})["validation"]
    print(f"Total validation samples: {len(dataset)}")
    
    # Generation config
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # Evaluate
    correct = 0
    total = 0
    unknown = 0
    
    print("\nStarting evaluation...")
    for example in tqdm(dataset, desc="Evaluating"):
        passage = example["passage"]
        question = example["question"]
        answer = example["answer"]
        gold = 1 if answer else 0
        
        pred = predict_boolq_with_generate(
            model, tokenizer, passage, question, device, MAX_LENGTH, gen_kwargs
        )
        if pred == -1:
            unknown += 1
        correct += int(pred == gold)
        total += 1
    
    accuracy = correct / total if total else 0.0
    print(f"\n===== Evaluation Results =====")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Unknown: {unknown}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()

