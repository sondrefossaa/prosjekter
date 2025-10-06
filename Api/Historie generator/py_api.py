#key = 56bb708f468b49aea852b03ab56a7950

from openai import OpenAI
import requests
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
base_url = "https://api.aimlapi.com/v1"
api_key = os.getenv('API_KEY')
context = []
system_prompt = """You are a master storyteller AI specializing in creating engaging, coherent, and imaginative narratives. Your purpose is to generate complete short stories based on user prompts while adhering to these guidelines:

1. **Story Structure**: Always follow a narrative arc with exposition, rising action, climax, falling action, and resolution.

2. **Character Development**: Create multidimensional characters with distinct personalities, motivations, and flaws.

3. **Setting**: Establish vivid, immersive worlds using sensory details and atmospheric descriptions.

4. **Genre Adaptation**: Tailor your writing style to match the requested genre (fantasy, sci-fi, romance, mystery, etc.).

5. **Length Management**: Respond in about 3 sentences, you can use abit more if you need to describe an environment or a conversation but never over 5 sentences.

6. **Originality**: Avoid clichés and overused tropes unless specifically requested. Offer fresh perspectives.

7. **Emotional Resonance**: Incorporate emotional depth that connects readers to characters and events.

8. **Pacing**: Balance description with action to maintain reader engagement throughout.

9. **Theme Integration**: Weave underlying themes throughout the narrative when appropriate.

10. **Language Quality**: Use rich vocabulary, varied sentence structures, and appropriate literary devices.
11. the most important is that you continue the story from the prompt
"""
user_prompt = "I walk into a bar and see a pretty woman"

api = OpenAI(api_key=api_key, base_url=base_url)


def generate_story_AIML():
    completion = api.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    response = completion.choices[0].message.content

    print("User:", user_prompt)
    print("AI:", response)
    return response

def generate_story_HF(prompt, master_prompt=None):
    client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:nebius",
        messages=[
            {
                "role": "user",
                "content": f"{master_prompt}\n\n{context}\n\nUser request: {prompt}"
            }
        ],
    )
    text = completion.choices[0].message.content
    context.append(text)
    print(text)
    return text
def generate_and_display_image(prompt, api_token=None, system_prompt=None):
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """
        You are an expert AI image generator. Create highly detailed, visually stunning images 
        based on the user's prompt. Focus on:
        - Photorealistic quality when appropriate
        - Proper lighting and composition
        - Vivid colors and contrast
        - Attention to fine details
        - Artistic interpretation when requested
        """
    
    # Combine system prompt with user prompt for better guidance
    enhanced_prompt = f"{system_prompt}\n\nUser request: {prompt}"
    
    client = InferenceClient(
        provider="fal-ai",
        api_key="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",
    )

    # Use the enhanced prompt for image generation
    image = client.text_to_image(
        enhanced_prompt,
        model="black-forest-labs/FLUX.1-Krea-dev",
    )
    image.show()
    
    return image


if __name__ == "__main__":
    #main()
        # Get your API token from https://huggingface.co/settings/tokens
    API_TOKEN = "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"  # Replace with your token
    
    # Generate and display image
    #story = generate_story_AIML()
    story = generate_story_HF("I walk into a bar an see a pretty woman", system_prompt)
    print(story)
    generated_image = generate_and_display_image(
        story,
        api_token=API_TOKEN
    )
    
