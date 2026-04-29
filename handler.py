import runpod
import torch
import base64
import io
import logging
import time
import re
import os
import json
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Network Volume Cache Setup (must be BEFORE model load)
os.makedirs("/runpod-volume/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/runpod-volume/hf_cache"
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
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
logging.info("Model loaded successfully.")


# Default Safety Policy
DEFAULT_POLICY = """Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm, or Cruelty: 
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content: 
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content: 
Should not:
- Contain visible genitalia or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning: 
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

O6: Weapons or Substance Abuse: 
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

O7: Self-Harm: 
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty: 
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies: 
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}
"""


# Response Parser
def parse_response(text: str) -> dict:
    # Primary: extract JSON from model output
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "rating": parsed.get("rating", "Unknown"),
                "category": parsed.get("category", "NA: None applying"),
                "rationale": parsed.get("rationale", text.strip())
            }
        except json.JSONDecodeError:
            pass

    # Fallback: keyword scan
    if re.search(r'\bunsafe\b', text, re.IGNORECASE):
        rating = "Unsafe"
    elif re.search(r'\bsafe\b', text, re.IGNORECASE):
        rating = "Safe"
    else:
        rating = "Unknown"

    return {
        "rating": rating,
        "category": "NA: None applying",
        "rationale": text.strip()
    }


# RunPod Handler
def handler(job):
    logging.info("New job received.")
    job_input = job.get("input", {})

    if "image_base64" not in job_input:
        logging.error("Missing 'image_base64' in input.")
        return {"error": "Missing 'image_base64' in input"}

    image_b64 = job_input["image_base64"]
    policy = job_input.get("policy", DEFAULT_POLICY)

    # Strip data-URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    # Decode Base64 → PIL Image
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logging.info(f"Image decoded. Size: {image.size}")
    except Exception as e:
        logging.error(f"Failed to decode image: {e}")
        return {"error": f"Failed to decode image_base64: {str(e)}"}

    # Build conversation — matches official model card exactly
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": policy},
            ],
        }
    ]

    # Prepare inputs — two-step as per official model card
    try:
        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        )
        # DTYPE FIX: convert float32 → float16 to match model
        # int tensors like input_ids are left untouched
        inputs = {
            k: (v.to(device).half() if v.dtype == torch.float32 else v.to(device))
            for k, v in inputs.items()
        }
    except Exception as e:
        logging.error(f"Failed to prepare inputs: {e}")
        return {"error": f"Failed to prepare inputs: {str(e)}"}

    # Run Inference
    try:
        logging.info("Running inference...")
        stime = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                use_cache=True,
            )

        decoded = processor.decode(output[0], skip_special_tokens=True)
        logging.info(f"Inference done in {time.perf_counter() - stime:.2f}s")
        logging.info(f"Raw output: {decoded}")

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return {"error": f"Inference failed: {str(e)}"}

    # Parse and return
    result = parse_response(decoded)

    return {
        "rating": result["rating"],
        "category": result.get("category", "NA: None applying"),
        "rationale": result["rationale"],
        "raw_output": decoded
    }


# IMPORTANT: This exact line is required by RunPod's GitHub scanner
runpod.serverless.start({"handler": handler})