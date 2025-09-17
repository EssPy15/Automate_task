# llm_client_local.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json

# Example model that runs on CPU reasonably: google/flan-t5-small (text2text)
MODEL = "google/flan-t5-small"

# Create pipeline (auto-downloads model locally)
pipe = pipeline("text2text-generation", model=MODEL)

def extract_event_info(email_text: str):
    system_prompt = "Extract assignment/test info and output exactly a JSON object with keys: type, course, date, details.\n\n"
    full_prompt = system_prompt + email_text
    result = pipe(full_prompt, max_new_tokens=128, do_sample=False)
    # result is a list of dicts [{ 'generated_text': '...'}]
    text = result[0]["generated_text"]
    # try to parse JSON out of model output:
    try:
        obj = json.loads(text)
        return obj
    except Exception:
        # Fallback: return raw string so your planner can handle it
        return text
