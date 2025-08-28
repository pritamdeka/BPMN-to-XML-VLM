
import os, time, json, re
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types

from src.utils import extract_xml_from_markdown, save_metrics, save_metrics_csv

class GeminiRedrawerPipeline:
    def __init__(self, api_key, prompt_file, image_folder, output_folder="gemini_outputs", model="gemini-2.5-flash"):
        self.api_key = api_key
        self.prompt_text = self._load_prompt(prompt_file)
        self.prompt_file = prompt_file
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.model = model
        os.makedirs(self.output_folder, exist_ok=True)
        self.client = genai.Client(api_key=self.api_key)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images_processed = 0
        self.total_processing_time = 0.0
        self.detailed_logs = []

    def _load_prompt(self, prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def extract_bpmn_output(self, image_path):
        image = Image.open(image_path)
        start = time.time()
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert BPMN image analyst with more than 30 years of experience.",
                    thinking_config=types.ThinkingConfig(thinking_budget=512)
                ),
                contents=[image, self.prompt_text]
            )
            dur = time.time() - start

            # usage fields may evolve; guard with getattr
            input_tokens = getattr(getattr(resp, "usage", None), "input_tokens", None) or 0
            output_tokens = getattr(getattr(resp, "usage", None), "output_tokens", None) or 0
            total_tokens = input_tokens + output_tokens

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_images_processed += 1
            self.total_processing_time += dur

            content_text = getattr(resp, "text", "") or ""
            self.detailed_logs.append({
                "model": self.model,
                "prompt_file": self.prompt_file,
                "prompt_length_chars": len(self.prompt_text),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens_for_call": total_tokens,
                "response_content_length_chars": len(content_text.strip()),
                "processing_time_sec": dur,
                "success": True
            })
            return extract_xml_from_markdown(content_text)
        except Exception as e:
            dur = time.time() - start
            self.detailed_logs.append({
                "model": self.model,
                "prompt_file": self.prompt_file,
                "prompt_length_chars": len(self.prompt_text),
                "error": str(e),
                "processing_time_sec": dur,
                "success": False
            })
            raise

    def process_single_image(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            bpmn_output = self.extract_bpmn_output(image_path)
            out_path = os.path.join(self.output_folder, f"{base_name}.bpmn")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(bpmn_output)
            tqdm.write(f"✅ BPMN saved: {out_path}")
        except Exception as e:
            tqdm.write(f"❌ Error processing {image_path}: {e}")

    def process_folder(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in tqdm(image_files, desc="Processing BPMN Images with Gemini"):
            self.process_single_image(os.path.join(self.image_folder, image_file))

    def calculate_metrics(self):
        total_tokens = self.total_input_tokens + self.total_output_tokens
        m = {
            "model_used": self.model,
            "prompt_file": self.prompt_file,
            "prompt_text": self.prompt_text,
            "total_images_processed": self.total_images_processed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "overall_total_tokens": total_tokens,
            "total_processing_time_seconds": self.total_processing_time,
        }
        if self.total_images_processed:
            m["avg_input_tokens_per_image"] = self.total_input_tokens / self.total_images_processed
            m["avg_output_tokens_per_image"] = self.total_output_tokens / self.total_images_processed
            m["avg_total_tokens_per_image"] = total_tokens / self.total_images_processed
            m["avg_processing_time_per_image_seconds"] = self.total_processing_time / self.total_images_processed
        else:
            m["avg_input_tokens_per_image"] = 0
            m["avg_output_tokens_per_image"] = 0
            m["avg_total_tokens_per_image"] = 0
            m["avg_processing_time_per_image_seconds"] = 0
        return m

    def save_detailed_logs(self, log_file="gemini_token_log.json"):
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.detailed_logs, f, indent=4)
        print(f"Detailed logs saved to {log_file}")
