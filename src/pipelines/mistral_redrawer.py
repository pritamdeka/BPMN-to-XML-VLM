
import os, time, base64, json, re
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from mistralai import Mistral

from src.utils import extract_xml_from_markdown, save_metrics, save_metrics_csv

class MistralBPMNRedrawerPipeline:
    def __init__(self, api_key, prompt_file, output_folder="mistral_outputs", model="mistral-small-2503"):
        self.api_key = api_key
        self.prompt_text = self._load_prompt(prompt_file)
        self.prompt_file = prompt_file
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.model = model
        self.client = Mistral(api_key=self.api_key)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_images_processed = 0
        self.total_processing_time = 0.0
        self.detailed_logs = []

    def _load_prompt(self, prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _encode_image_base64(self, image: Image.Image, max_size=(10000, 10000)):
        if image.width > max_size[0] or image.height > max_size[1]:
            image = image.copy()
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def extract_bpmn_output(self, image: Image.Image, image_file_name: str):
        encoded = self._encode_image_base64(image)
        start = time.time()
        try:
            resp = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": self.prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ]}
                ],
                max_tokens=10000,
            )
            dur = time.time() - start
            usage = resp.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_images_processed += 1
            self.total_processing_time += dur

            content = resp.choices[0].message.content.strip()
            self.detailed_logs.append({
                "image_file": image_file_name,
                "model": self.model,
                "prompt_file": self.prompt_file,
                "prompt_length_chars": len(self.prompt_text),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens_for_call": total_tokens,
                "response_content_length_chars": len(content),
                "processing_time_sec": dur,
                "success": True
            })
            return content
        except Exception as e:
            dur = time.time() - start
            self.detailed_logs.append({
                "image_file": image_file_name,
                "model": self.model,
                "prompt_file": self.prompt_file,
                "prompt_length_chars": len(self.prompt_text),
                "error": str(e),
                "processing_time_sec": dur,
                "success": False
            })
            raise

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            bpmn_output = self.extract_bpmn_output(image, os.path.basename(image_path))
            clean_bpmn = extract_xml_from_markdown(bpmn_output)
            out_path = os.path.join(self.output_folder, f"{base_name}.bpmn")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(clean_bpmn)
            tqdm.write(f"✅ BPMN saved: {out_path}")
        except Exception as e:
            tqdm.write(f"❌ Error processing {image_path}: {e}")

    def process_folder(self, image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in tqdm(image_files, desc="Processing BPMN Images with Mistral"):
            self.process_image(os.path.join(image_folder, image_file))

    def calculate_metrics(self):
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        m = {
            "model_used": self.model,
            "prompt_file": self.prompt_file,
            "prompt_text": self.prompt_text,
            "total_images_processed": self.total_images_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "overall_total_tokens": total_tokens,
            "total_processing_time_seconds": self.total_processing_time,
        }
        if self.total_images_processed > 0:
            m["avg_prompt_tokens_per_image"] = self.total_prompt_tokens / self.total_images_processed
            m["avg_completion_tokens_per_image"] = self.total_completion_tokens / self.total_images_processed
            m["avg_total_tokens_per_image"] = total_tokens / self.total_images_processed
            m["avg_processing_time_per_image_seconds"] = self.total_processing_time / self.total_images_processed
        else:
            m["avg_prompt_tokens_per_image"] = 0
            m["avg_completion_tokens_per_image"] = 0
            m["avg_total_tokens_per_image"] = 0
            m["avg_processing_time_per_image_seconds"] = 0
        return m

    def save_detailed_logs(self, log_file="mistral_token_log.json"):
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.detailed_logs, f, indent=4)
        print(f"Detailed logs saved to {log_file}")
