
import os
import json
import requests
from google import genai
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1")
REFINE_MODEL = os.getenv("REFINE_MODEL", "openai/gpt-5-pro")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=GOOGLE_API_KEY)
PRIMARY_MODEL = "gemini-2.5-flash"

import codecs

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        print(f"UTF-8 decoding failed for {path}. Trying with 'latin-1'...")
        try:
            with open(path, "r", encoding="latin-1") as f:
                return json.load(f)
        except Exception as e:
            print(f" Failed to read {path} with 'latin-1' encoding: {e}")
            raise
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed for {path}: {e}")
        # Attempt to read partially if the error is unexpected end of data
        if "unexpected end of data" in str(e):
             try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Find the last valid JSON object or array
                    last_brace = content.rfind('}')
                    last_bracket = content.rfind(']')
                    if last_brace > last_bracket:
                        content = content[:last_brace+1]
                    elif last_bracket > last_brace:
                         content = content[:last_bracket+1]
                    else:
                         raise e # Re-raise if no valid end found

                    return json.loads(content)
             except Exception as inner_e:
                print(f" Failed to partially read and decode {path}: {inner_e}")
                raise e # Re-raise original error if partial read fails
        else:
            raise # Re-raise other JSONDecodeErrors
    except Exception as e:
        print(f" An unexpected error occurred while reading {path}: {e}")
        raise

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_openrouter_connection():
    print("Checking OpenRouter connectivity...")
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            print(" OpenRouter connection successful.")
            return True
        else:
            print(f"OpenRouter connection returned {response.status_code}: {response.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to OpenRouter: {e}")
    return False

class DictListProcessor:
    """Extracts unique field values from a list of dicts."""
    def __init__(self, data):
        self.data = data
        self._unique_values = self._extract_unique_values()

    def _extract_unique_values(self):
        if not self.data:
            return {}
        keys = [k for k in self.data[0].keys() if k != "image"]
        result = {}
        for key in keys:
            vals = set()
            for d in self.data:
                v = d.get(key)
                if isinstance(v, list):
                    vals.update(v)
                else:
                    vals.add(v)
            result[key] = list(vals)
        return result

    def get_unique_values(self):
        return self._unique_values

def cluster_with_gemini(entity, values):
    """Cluster label variants using Gemini 2.5 Flash."""
    prompt = f"""
You are a data labeling assistant.
Cluster the following unique values for '{entity}' into groups of similar meaning.

Values:
{values}

Return ONLY valid JSON in this format:
{{
  "blue": ["blue", "navy", "sky blue", "آبی"],
  "unrelated": ["unknown"]
}}
"""
    try:
        resp = client.models.generate_content(model=PRIMARY_MODEL, contents=prompt).text
        start, end = resp.find("{"), resp.rfind("}")
        if start != -1 and end != -1:
            return json.loads(resp[start:end + 1])
    except Exception as e:
        print(f"Gemini clustering failed for {entity}: {e}")
    return {}

def call_gpt5_pro(prompt, retries=3, delay=5):
    """Call GPT-5 Pro via OpenRouter with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": REFINE_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a precise data-refinement assistant that outputs valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1500,
            }
            resp = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            print(f" GPT-5 request failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(" All retry attempts failed.")
                raise

def refine_with_gpt5(entity, clusters):
    """Refine and merge Gemini clusters using GPT-5 Pro."""
    prompt = f"""
You are refining label clusters for the entity '{entity}'.

Current clusters:
{json.dumps(clusters, ensure_ascii=False, indent=2)}

Rules:
- Merge clusters with the same semantic meaning, not too general and not too detailed.
- Rename clusters to clear English labels.
- Keep unrelated or ambiguous items in "unrelated".
- Return ONLY valid JSON (no text outside JSON).
"""
    try:
        reply = call_gpt5_pro(prompt)
        start, end = reply.find("{"), reply.rfind("}")
        if start != -1 and end != -1:
            return json.loads(reply[start:end + 1])
    except Exception as e:
        print(f" GPT-5 refinement failed for {entity}: {e}")
    return clusters

def run_pipeline(input_path="cleaned_dict.json", output_path="final_clusters.json"):
    if not check_openrouter_connection():
        print(" OpenRouter connectivity check failed. Exiting.")
        return

    data = read_json(input_path)
    processor = DictListProcessor(data)
    results = {}

    for entity, values in processor.get_unique_values().items():
        print(f"\n🔹 Processing '{entity}' with {len(values)} unique values...")
        first_pass = cluster_with_gemini(entity, values)
        refined = refine_with_gpt5(entity, first_pass)
        results[entity] = refined
        print(f" {entity} refinement complete.\n")

    write_json(output_path, results)
    print(f"\n All refined clusters saved to '{output_path}'")

if __name__ == "__main__":
    run_pipeline()

