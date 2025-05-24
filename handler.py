#!/usr/bin/env python
# Handler for RunPod Serverless Gemma 3
import os
import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

# ===== USER MODIFIABLE SETTINGS =====
# Get model ID from environment variable with fallback to default
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-4b-it")

# Maximum tokens to generate with fallback to default
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
# =====================================

# Set up Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load the model once at startup, outside of the handler
print(f"Loading model: {MODEL_ID}")
print(f"Default max tokens: {MAX_NEW_TOKENS}")

# Configure token parameters if provided
if HF_TOKEN:
    token_param = {"token": HF_TOKEN}
    print("Using configured Hugging Face token from environment variable")
else:
    token_param = {}
    print("No Hugging Face token provided (this will only work for non-gated models)")

# Configure quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **token_param,
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID, **token_param)

    print(f"Model loaded on {device}")
except Exception as e:
    if not HF_TOKEN:
        print("ERROR: Failed to load model. This may be a gated model that requires a token.")
    raise e

print("Model and processor loaded and ready for inference")


def invoke(image_data=None, prompt='', max_new_tokens=MAX_NEW_TOKENS):
    """Generate a caption for the given image."""
    try:
        # Create messages for the model with custom prompt
        if image_data:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_data}
            ]
        else:
            content = [
                {"type": "text", "text": prompt}
            ]

        messages = [{"role": "user", "content": content}]

        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Track input length to extract only new tokens
        input_len = inputs["input_ids"].shape[-1]

        # Generate caption
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        # Extract only the newly generated tokens
        generated_tokens = outputs[0][input_len:]

        # Decode the caption
        answer = processor.decode(generated_tokens, skip_special_tokens=True)

        # Ensure caption is a single line
        return answer.replace('\n', ' ').strip()

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"Error: {str(e)}\n{traceback_str}"


def handler(job: dict):
    """
    This is the handler function that will be called by the serverless worker.
    Job input format:
    {
        "image": "base64 encoded image or URL",
        "prompt": "Optional custom prompt for captioning",
        "max_new_tokens": 256  # Optional, defaults to 256
    }
    """
    job_input = job["input"]

    # Get the prompt (optional, use default if not provided)
    prompt = job_input["prompt"]
    max_new_tokens = job_input.get("max_new_tokens", MAX_NEW_TOKENS)


    if image_input := job_input.get("image"): # Handle the image (base64, URL, or file path)
        try:
            # Case 1: Base64 encoded image
            if isinstance(image_input, str) and image_input.startswith("data:image"):
                # Extract base64 part after the comma
                base64_data = image_input.split(",")[1]
                image_data = Image.open(io.BytesIO(base64.b64decode(base64_data)))

            # Case 2: Pure base64 string (without data URI prefix)
            elif isinstance(image_input, str) and len(image_input) > 100:
                try:
                    image_data = Image.open(io.BytesIO(base64.b64decode(image_input)))
                except Exception:
                    # If not a valid base64, try as URL or file path
                    if image_input.startswith(('http://', 'https://')):
                        # It's a URL, we need to download it
                        import requests
                        response = requests.get(image_input, stream=True)
                        response.raise_for_status()  # Will raise an exception for HTTP errors
                        image_data = Image.open(io.BytesIO(response.content))
                    else:
                        # Assume it's a file path
                        image_data = Image.open(image_input)

            # Case 3: URL starting with http:// or https://
            elif isinstance(image_input, str) and image_input.startswith(('http://', 'https://')):
                # It's a URL, we need to download it
                import requests
                response = requests.get(image_input, stream=True)
                response.raise_for_status()  # Will raise an exception for HTTP errors
                image_data = Image.open(io.BytesIO(response.content))

            # Case 4: Local file path
            elif isinstance(image_input, str):
                # Assume it's a file path
                image_data = Image.open(image_input)

            else:
                return {"error": "Invalid image format. Please provide a base64 encoded image, URL, or file path."}

            # Convert to RGB mode to ensure compatibility
            image_data = image_data.convert("RGB")

            # Process the image to get the caption
            return invoke(image_data, prompt, max_new_tokens)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return {"error": f"Error processing image: {str(e)}", "traceback": error_trace}
    else:
        return invoke(None, prompt, max_new_tokens) # Handle the text prompt

# Start the serverless function
runpod.serverless.start({"handler": handler})
