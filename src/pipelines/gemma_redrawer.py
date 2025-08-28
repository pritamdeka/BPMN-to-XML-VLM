
import os, json, time, re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.utils import extract_xml_from_markdown

Image.MAX_IMAGE_PIXELS = None
torch.set_float32_matmul_precision("high")

class GemmaRedrawerPipeline:
    def __init__(self, model_id, prompt_file, image_folder, output_folder="gemma_outputs"):
        self.model_id = model_id
        self.prompt_text = self._load_prompt(prompt_file)
        self.prompt_file = prompt_file
        self.image_folder = image_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.detailed_logs = []
        self.total_images_processed = 0
        self.total_processing_time = 0.0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0

    def _load_prompt(self, prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def extract_bpmn_output(self, image_path, image_file_name):
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": self.prompt_text}
            ]}
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32))
        input_len = inputs["input_ids"].shape[-1]
        prompt_tokens = int(input_len)

        start = time.time()
        with torch.no_grad():
            generation = self.model.generate(**inputs, max_new_tokens=10000, do_sample=False)
        dur = time.time() - start

        output_tokens = generation[0][input_len:]
        completion_tokens = int(output_tokens.shape[0])
        output_text = self.processor.decode(output_tokens, skip_special_tokens=True)
        total_tokens = prompt_tokens + completion_tokens

        self.detailed_logs.append({
            "image_file": image_file_name,
            "model": self.model_id,
            "prompt_file": self.prompt_file,
            "prompt_length_chars": len(self.prompt_text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens_for_call": total_tokens,
            "response_content_length_chars": len(output_text.strip()),
            "processing_time_sec": dur,
            "success": True
        })
        self.total_images_processed += 1
        self.total_processing_time += dur
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        return extract_xml_from_markdown(output_text)

    def process_single_image(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_file_name = os.path.basename(image_path)
        try:
            bpmn_output = self.extract_bpmn_output(image_path, image_file_name)
            out_path = os.path.join(self.output_folder, f"{base_name}.bpmn")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(bpmn_output)
            tqdm.write(f"✅ BPMN saved: {out_path}")
        except Exception as e:
            self.detailed_logs.append({
                "image_file": image_file_name,
                "model": self.model_id,
                "prompt_file": self.prompt_file,
                "prompt_length_chars": len(self.prompt_text),
                "error": str(e),
                "processing_time_sec": 0.0,
                "success": False
            })
            tqdm.write(f"❌ Error processing {image_path}: {e}")

    def process_folder(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in tqdm(image_files, desc="Processing BPMN Images with Gemma"):
            self.process_single_image(os.path.join(self.image_folder, image_file))

    def calculate_metrics(self):
        total_tokens = self.total_prompt_tokens + self.total_output_tokens
        m = {
            "model_used": self.model_id,
            "prompt_file": self.prompt_file,
            "prompt_text": self.prompt_text,
            "total_images_processed": self.total_images_processed,
            "total_prompt_tokens": int(self.total_prompt_tokens),
            "total_completion_tokens": int(self.total_output_tokens),
            "overall_total_tokens": int(total_tokens),
            "total_processing_time_seconds": self.total_processing_time,
        }
        if self.total_images_processed:
            m["avg_prompt_tokens_per_image"] = self.total_prompt_tokens / self.total_images_processed
            m["avg_completion_tokens_per_image"] = self.total_output_tokens / self.total_images_processed
            m["avg_total_tokens_per_image"] = total_tokens / self.total_images_processed
            m["avg_processing_time_per_image_seconds"] = self.total_processing_time / self.total_images_processed
        else:
            m["avg_prompt_tokens_per_image"] = 0
            m["avg_completion_tokens_per_image"] = 0
            m["avg_total_tokens_per_image"] = 0
            m["avg_processing_time_per_image_seconds"] = 0
        return m

    def save_detailed_logs(self, log_file="gemma_token_log.json"):
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.detailed_logs, f, indent=4)
        print(f"Detailed logs saved to {log_file}")
