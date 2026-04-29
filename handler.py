import runpod
import torch
import base64
import io
import logging
import time
import re
import os
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Network Volume Cache Setup (must be BEFORE transformers model load)
os.makedirs("/runpod-volume/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/runpod-volume/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/runpod-volume/hf_cache"

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Load 
MODEL_NAME = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"

logging.info(f"Loading LlavaGuard model on {device}...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.eval()
model.to(device)

if device == "cuda":
    logging.info(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Running on CPU (no GPU available)")

logging.info("Model loaded successfully.")

#  Default Safety Policy
DEFAULT_POLICY = """Provide a safety assessment for the content provided by the user. Determine compliance or violation based on the following policy:
O1: Hate, Humiliation, Harassment: Should not contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, gender, sexual orientation, disability, or physical appearance.
O2: Violence, Harm, or Cruelty: Should not contain content of people or animals being maimed, brutalized, physically abused, or tortured.
O3: Sexual Content: Should not contain sexually explicit material (pornography).
O4: Nudity Content: Should not contain visible genitalia.
O5: Criminal Planning: Should not contain content of crimes being planned or committed.
O6: Weapons or Substance Abuse: Should not contain content of illegal weapons or illegal substance abuse.
O7: Self-Harm: Should not contain content that displays, promotes, or provides instructions for self-harm.
O8: Animal Cruelty: Should not contain content of animals being harmed or killed.
O9: Disasters or Emergencies: Should not contain content that exploits or makes fun of victims of disasters.
Assessment: Provide your safety assessment for the image in the above conversation. Answer with 'Safe' or 'Unsafe', followed by a brief explanation."""


# Response Parser 
def parse_response(text: str) -> dict:
    """Extract rating and rationale from model output."""
    # Try to find ASSISTANT turn first
    match = re.search(r"ASSISTANT:\s*(Safe|Unsafe)(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        rating = match.group(1).capitalize()
        rationale = match.group(2).strip()
        return {"rating": rating, "rationale": rationale}

    # Fallback: scan full text for Safe/Unsafe
    if re.search(r"\bunsafe\b", text, re.IGNORECASE):
        rating = "Unsafe"
    elif re.search(r"\bsafe\b", text, re.IGNORECASE):
        rating = "Safe"
    else:
        rating = "Unknown"

    return {"rating": rating, "rationale": text.strip()}


# RunPod Handler 
def handler(job):
    """RunPod handler — called once per job."""
    logging.info("New job received.")
    job_input = job.get("input", {})

    # Validate Input 
    if "image_base64" not in job_input:
        logging.error("Missing 'image_base64' in input.")
        return {"error": "Missing 'image_base64' in input"}

    image_b64 = job_input["image_base64"]
    policy = job_input.get("policy", DEFAULT_POLICY)

    # Strip data-URI prefix if present e.g. "data:image/jpeg;base64,..."
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    #  Decode Base64 → PIL Image
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logging.info(f"Image decoded successfully. Size: {image.size}")
    except Exception as e:
        logging.error(f"Failed to decode image: {e}")
        return {"error": f"Failed to decode image_base64: {str(e)}"}

    #  Build Prompt 
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": policy},
            ],
        }
    ]

    try:
        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
    except Exception as e:
        logging.error(f"Failed to build prompt: {e}")
        return {"error": f"Failed to build prompt: {str(e)}"}

    #  Run Inference
    try:
        logging.info("Running LlavaGuard inference...")
        stime = time.perf_counter()

        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        hyperparameters = {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "num_beams": 2,
            "use_cache": True,
        }

        with torch.no_grad():
            output = model.generate(**inputs, **hyperparameters)

        decoded = processor.decode(output[0], skip_special_tokens=True)
        etime = time.perf_counter()
        logging.info(f"Inference completed in {etime - stime:.2f}s")
        logging.info(f"Raw output: {decoded}")

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return {"error": f"Inference failed: {str(e)}"}

    #  Parse & Return 
    result = parse_response(decoded)

    return {
        "rating": result["rating"],        # "Safe" or "Unsafe"
        "rationale": result["rationale"],  # explanation
        "raw_output": decoded              # full model output for debugging
    }


# IMPORTANT: This exact line is required by RunPod's GitHub scanner
runpod.serverless.start({"handler": handler})