import {api_key} from './config.js'

import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(api_key);

const image = await client.textToImage({
    provider: "fal-ai",
    model: "black-forest-labs/FLUX.1-Krea-dev",
    inputs: "Astronaut riding a horse",
    parameters: { num_inference_steps: 5 },
});

// Create a URL for the blob
const imageUrl = URL.createObjectURL(image);

// Display in an img element
const imgElement = document.createElement('img');
imgElement.src = imageUrl;
document.body.appendChild(imgElement);

// Don't forget to revoke the URL when done to free memory
// URL.revokeObjectURL(imageUrl); // Call this when you no longer need the image
/// Use the generated image (it's a Blob)