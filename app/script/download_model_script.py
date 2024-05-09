from transformers import GPT2LMHeadModel, GPT2Tokenizer


def download_and_save_model(model_name, save_directory):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
