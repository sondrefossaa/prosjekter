const api_key = "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"
/* async function query(data) {
	const response = await fetch(
		"https://router.huggingface.co/v1/chat/completions",
		{
			headers: {
				Authorization: `Bearer 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`,
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query({ 
    messages: [
        {
            role: "user",
            content: "What is the capital of France?",
        },
    ],
    model: "meta-llama/Llama-3.1-8B-Instruct:cerebras",
}).then((response) => {
    console.log(JSON.stringify(response));
}); */

/* async function query(data) {
	const response = await fetch(
		"https://router.huggingface.co/fal-ai/fal-ai/flux/krea",
		{
			headers: {
				Authorization: `Bearer ${process.env.HF_TOKEN}`,
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.blob();
	return result;
}


query({     sync_mode: true,
    prompt: "\"Astronaut riding a horse\"", }).then((response) => {
    // Use image
    result.show()
}); */
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(api_key);

const image = await client.textToImage({
    provider: "fal-ai",
    model: "black-forest-labs/FLUX.1-Krea-dev",
	inputs: "Astronaut riding a horse",
	parameters: { num_inference_steps: 5 },
});
/// Use the generated image (it's a Blob)