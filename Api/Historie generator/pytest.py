import os
os.environ["HF_HOME"] = r"E:\personlige ting\koding\ai modeller"

from transformers import pipeline
import torch_directml

device = torch_directml.device()
generator = pipeline('text-generation', model='openai-community/gpt2', device=device)

results = generator("Hello, I'm a language model", truncation=True, num_return_sequences=3)

# Method 1: Print each generated text with numbering
print("Generated texts:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['generated_text']}")
    print()  # Add empty line between results

print("-" * 50)

# Method 2: Just the text without numbering
print("Clean output:")
for result in results:
    print(result['generated_text'])
    print()

print("-" * 50)

# Method 3: Access specific result
print("First result only:")
print(results[0]['generated_text'])