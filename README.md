
# GPT-2 Text Generation Script

This script uses the `transformers` library to generate text based on a prompt provided via the command line. It utilizes the pre-trained GPT-2 model from OpenAI to perform text generation tasks.

## Model Parameters
- **gpt2**: 117M parameters, requires 8 GB GPU or 2-3 GB RAM
- **gpt2-medium**: 345M parameters, requires 12 GB GPU or 4-6 GB RAM
- **gpt2-large**: 774M parameters, requires 16 GB GPU or 8 GB RAM
- **gpt2-xl**: 1.5B parameters, requires 24 GB GPU or 16 GB RAM

## Dependencies
To run this script, you need the following Python libraries:
- `transformers==4.40.2`
- `torch==2.3.0`
- `torchvision==0.18.0`
- `torchaudio==2.3.0`

```
pip install -r requirements.txt
```

## How It Works
1. **Input Handling**: The script expects a single command line argument, which is the text prompt for the model. If the prompt is not provided, the script will exit with a usage message.
2. **Model and Tokenizer Setup**: It loads the `GPT2LMHeadModel` and `GPT2Tokenizer` from the `transformers` library. The model and tokenizer are used to process the input prompt and generate text.
3. **Token Encoding**: The provided text prompt is encoded into tokens using the tokenizer.
4. **Text Generation**: These tokens are then fed to the model which generates a sequence of tokens as output.
5. **Decoding and Output**: The output tokens are decoded back into human-readable text and printed along with the processing time.

This script provides a simple interface to harness the power of GPT-2 for generating text.

## Execution
Run the script from the command line:
```
python script.py 'your prompt'
```

Note: Ensure that the working environment has enough RAM or GPU resources based on the GPT-2 model variant used.
