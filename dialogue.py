#!/usr/bin/env python3

import argparse
import sys
import time
from typing import List, Optional
import re

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp python not installed. Please install it using:")
    print("pip install llama-cpp-python")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description = 'Local LLM dialogue system')
    
    parser.add_argument('--model', type = str, required = True, 
                        help = 'Path to the GGUF model file')
    
    parser.add_argument('--n_ctx', type = int, default = 2048,
                        help = 'Context size for the model')

    parser.add_argument('--n_gpu_layers', type = int, default = 0,
                        help = "Number of layers to offload your GPU")
    
    parser.add_argument('--interactive', action = 'store_true',
                        help = 'Start an interactive dialogue system')

    parser.add_argument('--question', type = str,
                        help = 'Question to ask (non-interactive mode)')

    parser.add_argument('--max_tokens', type = int, default = 512,
                        help = "Maximum number of tokens to generate")

    parser.add_argument('--temp', type = float, default = 0.3,
                        help = 'Temperature for sampling')
    
    return parser.parse_args()

def load_model(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0) -> Llama:
    """Load the LLM model"""
    """The -> variable_name syntax in a Python function definition is a type hint
    that indicates the return type of the function. 
    
    It doesn't refer to an actual variable name, but rather specifies what type of value the function is expected to return"""

    """The type hints are optional and don't affect how the code runs
    - Python will still execute the function normally even if the actual return type doesn't match the hint."""

    """However, they're useful for Documentation"""

    print(f"Loading model from {model_path}...")

    try:
        model = Llama(model_path = model_path, n_ctx = n_ctx,
                      n_gpu_layers = n_gpu_layers, verbose = False)

        """The verbose parameter controls console output and logging during model initialization and operation"""

        """When verbose=True (default):
        
        Prints loading progress and status messages to the console during model initialization
        
        Shows detailed information about model loading, memory allocation, and other internal operations
        
        Displays diagnostic information that can be helpful for debugging"""

        """When verbose=False:
        
        Suppresses most console output during model loading and operation
        
        Provides a "quiet" mode for cleaner program output
        
        Can affect console output behavior in multi-threaded applications"""

        """Here verbose = False for cleaner output"""

        print("Model loaded successfully!")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)



def generate_streaming_response(model: Llama, prompt: str,
                                max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a streaming response from the model."""

    full_response = ""

    print("\nModel is thinking...\n")

    stop_sequences = ["<|im_end|>", "<|endoftext|>", "\nYou:", "\n\nYou:", "\n用户:", "\n\n用户:"]

    try:
        # Primary streaming method
        response = model.create_completion(prompt, max_tokens = max_tokens,
                                           temperature = temperature, stop = stop_sequences,
                                           stream = True, echo = False)
        """When echo: bool = False (default): Response contains only the generated completion
        
            When echo: bool = True: Response includes the original prompt + the generated completion"""

        chunk_count = 0

        for chunk in response:
            chunk_count += 1

            if 'choices' in chunk and len(chunk['choices']) > 0:
                choice = chunk['choices'][0]

                # Handle different possible structures that will be appeared in the process of response

                text_chunk = ""

                if 'delta' in choice and 'content' in choice['delta']:
                    text_chunk = choice['delta']['content']
                
                elif 'text' in choice:
                    text_chunk = choice['text']
                
                elif 'content' in choice:
                    text_chunk = choice['content']

                if text_chunk:
                    # Very minimal filtering - only check for obvious unwanted content
                    if len(full_response + text_chunk) > 100 and contains_unwanted_content(full_response):
                        break

                    full_response += text_chunk
                    print(text_chunk, end = "", flush = True)

                if choice.get('finish_reason') in ['stop', 'length']:
                    break

        if chunk_count == 0:
            return generate_with_tokens(model, prompt, max_tokens, temperature, stop_sequences)

    except Exception as e:
        print(f"\nStreaming failed: {e}")
        print("Falling back to token generation...", flush = True)
        return generate_with_tokens(model, prompt, max_tokens, temperature, stop_sequences)

    # Clean up the response
    full_response = clean_response(full_response)

    print("\n")

    return full_response

def generate_with_tokens(model: Llama, prompt: str, max_tokens: int, temperature: float, stop_sequences: list) -> str:
    """Fallback token-by-token generation method"""
    full_response = ""

    try:
        tokens = model.tokenize(prompt.encode("utf-8"))

        generated_count = 0

        for token in model.generate(tokens, temp = temperature):
            text_chunk = model.detokenize([token]).decode("utf-8", errors = "ignore")

            temp_response = full_response + text_chunk
            should_stop = False

            # Check for stop sequences

            for stop_seq in stop_sequences:
                if stop_seq in temp_response:
                    clean_response = temp_response.split(stop_seq)[0]
                    remaining_chunk = clean_response[len(full_response):]

                    if remaining_chunk:
                        print(remaining_chunk, end = "", flush = True)

                        # flush parameter controls whether the output is immediately written to the terminal/file or if it's buffered

                        full_response = clean_response
                        should_stop = True
                        break

            if should_stop:
                break

            if len(temp_response) > 100 and contains_unwanted_content(temp_response):
                break

            full_response += text_chunk
            print(text_chunk, end = "", flush = True)
            generated_count += 1

            if generated_count >= max_tokens:
                break
    
    except Exception as e:
        print(f"Token generation error: {e}")
        return "I apologize, but I encountered an error while generating a response."

    return clean_response(full_response)

def contains_unwanted_content(text: str) -> bool:
    """Check if text contains unwanted content that should trigger stopping"""

    """The reason to add this function in the implementation is that 
    we would like to focus on English context but
    the LLM used in this project is qwen, which is developed in the Chinese-centric way
    
    So we need a function to surpress the irrelevant Chinese context to appear"""

    if len(text.strip()) < 80:
        # text.strip(): a Python string method that removes whitespace characters from both the beginning and end of a string

        return False
    
    specific_problems = [r'极坐标.*点P',  # Specific math problem
        r'选择题.*[A-D][.)].*[A-D][.)]',  # Multiple choice questions
        r'[\u4e00-\u9fff].*[\u4e00-\u9fff].*[\u4e00-\u9fff]',  # Multiple Chinese characters
        ]

    for pattern in specific_problems:
        if re.search(pattern, text):
            return True

    # re.search() scans through the entire string looking for the first location where the pattern matches
    """pattern: The regular expression pattern to search for (string or compiled regex object)
    
       string: The string to search in"""

    return False

def should_stop_generation(text: str) -> bool:
    """Additional checks to determine if generation should stop"""

    # Only apply after significant content

    if len(text.strip()) < 100:
        return False

    # Only stop for very obvious problems

    if re.search(r'极坐标.*点P.*距离', text):
        return True
    
    return False

def clean_response(text: str) -> str:
    """Clean and finalize the response"""

    text = text.strip()

    specific_unwanted = [
        r'极坐标.*',
        r'选择题.*',
    ]

    for pattern in specific_unwanted:
        match = re.search(pattern, text)
        if match:
            before_pattern = text[:match.start()].strip()

            """Returns an integer representing the index where the match begins
               Uses zero-based indexing (first character is position 0)
               Only works if a match is found """

            if len(before_pattern) > 20:
                text = before_pattern
                break

    return text.strip()

def build_chatml_prompt(history, user_input):
    prompt = "<|im_start|>system\nYou are a helpful AI assistant. Respond concisely and relevantly to the user's question. Always end your response with <|im_end|> and do not generate any additional content after that.<|im_end|>\n"

    for i, turn in enumerate(history):
        if i % 2 == 0:
            prompt += f"<|im_start|>user\n{turn}<|im_end|>\n"

        else:
            prompt += f"<|im_start|>assistant\n{turn}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    return prompt

def run_interactive_dialogue(model: Llama, max_tokens: int, temperature: float):
    """Run an interactive dialogue session"""

    print("\n=== Interactive LLM Dialogue Session ===")
    print("Type 'exit' or 'quit' to end the session.\n")

    history = []

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Ending dialogue session. Goodbye!")
            break

        prompt = build_chatml_prompt(history, user_input)
        print("AI:", end = "")

        response = generate_streaming_response(model = model, prompt = prompt,
                                                max_tokens = max_tokens, temperature = temperature)
        
        # Only keep the response up to the first <|im_end|>
        response = response.split("<|im_end|>")[0].strip()
        history.append(user_input)
        history.append(response)

        """Splitting the response text at the first occurrence of <|im_end|> token
           Taking only the part before that token ([0])
           Stripping any leading/trailing whitespace
           This prevents the model from generating content beyond its intended stopping point"""

        if len(history) > 20:
            history = history[-20:]
            # -20 here means return the last 20 items

def main():
    """Main function to run the dialogue program"""

    args = parse_arguments()

    # Load the model

    model = load_model(args.model, args.n_ctx, args.n_gpu_layers)

    """You can access the parsed arguments as attributes on this object, for example:
       args.model (the model path)
       args.n_ctx (context size)
       args.interactive (boolean flag)
       args.question (question string) """

    if args.interactive:
        run_interactive_dialogue(model, args.max_tokens, args.temp)

    elif args.question:
        prompt = f"You: {args.question}\nAI:"
        print(f"\nQuestion: {args.question}")
        print("AI:", end = "")

        generate_streaming_response(model = model, prompt = prompt,
                                    max_tokens = args.max_tokens, temperature = args.temp)

    else:
        print("Error: Either --interactive or --question must be specified")
        sys.exit(1)

if __name__ == "__main__":
    main()
