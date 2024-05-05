
import onnxruntime_genai as og
import argparse
import time
from pathlib import Path

# Get the directory of the current script
current_dir = Path(__file__).parent
# Go up one level and then into the 'directml' directory
DML_MODEL_FOLDER = str(current_dir.parent / 'directml'/'directml-int4-awq-block-128')

def generate_response( prompt, max_output_tokens, temperature):
    model = og.Model(DML_MODEL_FOLDER)
    tokenizer = og.Tokenizer(model)
    search_options = {
        'max_length': max_output_tokens,
        'temperature': temperature
    }

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    prompt = f'{chat_template.format(input=prompt)}'
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.try_use_cuda_graph_with_max_batch_size(1)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    response_tokens = []
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        response_tokens.append(new_token)

    response = tokenizer.decode(response_tokens)
    return response


def main(args):
    #model_path = args.model
    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue

        response = generate_response(text, args.max_output_tokens, args.temperature)
        print("Output:", response)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    #parser.add_argument('-m', '--model', type=str, default=DML_MODEL_FOLDER, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-l', '--max_output_tokens', type=int, default=100, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature to sample with')
    args = parser.parse_args()
    main(args)
