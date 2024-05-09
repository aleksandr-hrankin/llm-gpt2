import sys
import time

from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODEL_NAME = "gpt2-xl"


def get_current_str_time() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py 'your prompt'")
        return

    prompt = sys.argv[1]

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True)
    print(f"[{get_current_str_time()}] Inputs: {inputs}")

    start_time = time.time()
    print(f"[{get_current_str_time()}] Start processing")

    outputs = model.generate(
        inputs,
        do_sample=True,
        temperature=0.9,
        max_length=100
    )
    print(f"[{get_current_str_time()}] Outputs: {outputs}")

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[{get_current_str_time()}] End processing")

    end_time = time.time()
    print(f"[{get_current_str_time()}] Total processing time: {end_time - start_time:.2f} seconds")

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
