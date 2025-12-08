
import os, json, time, re, csv
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    # Fallback shim if the helper isn't installed; user should install qwen's utils.
    def process_vision_info(messages):
        # Minimal placeholder: no-op, relies on AutoProcessor handling image paths directly
        images = [c["image"] for m in messages for c in m["content"] if c.get("type") == "image"]
        return images, None

from src.utils import extract_xml_from_markdown

min_pixels = 256*28*28
max_pixels = 1280*28*28

class QwenVLRedrawerPipeline:
    def __init__(self, model_id, prompt_file, image_folder, output_folder="qwen_outputs", device=None):
        self.model_id = model_id
        self.prompt_text = self._load_prompt(prompt_file)
        self.prompt_file = prompt_file
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_folder, exist_ok=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=False
        )

        self.detailed_logs = []
        self.total_images_processed = 0
        self.total_processing_time = 0.0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0

    def _load_prompt(self, prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def extract_bpmn_output(self, image_path, image_file_name):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": self.prompt_text},
            ]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        input_len = inputs.input_ids.shape[-1]
        prompt_tokens = int(input_len)

        start = time.time()
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=10000)
        dur = time.time() - start

        gen_trim = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        output_tokens = int(gen_trim[0].shape[0])
        output_text = self.processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        total_tokens = prompt_tokens + output_tokens

        self.detailed_logs.append({
            "image_file": image_file_name,
            "model": self.model_id,
            "prompt_file": self.prompt_file,
            "prompt_length_chars": len(self.prompt_text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens_for_call": total_tokens,
            "response_content_length_chars": len(output_text.strip()),
            "processing_time_sec": dur,
            "success": True
        })
        self.total_images_processed += 1
        self.total_processing_time += dur
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += output_tokens

        return extract_xml_from_markdown(output_text)

    def process_single_image(self, image_path):
        base = os.path.splitext(os.path.basename(image_path))[0]
        try:
            bpmn = self.extract_bpmn_output(image_path, os.path.basename(image_path))
            out_path = os.path.join(self.output_folder, f"{base}.bpmn")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(bpmn)
            tqdm.write(f"✅ BPMN saved: {out_path}")
        except Exception as e:
            self.detailed_logs.append({
                "image_file": os.path.basename(image_path),
                "model": self.model_id,
                "prompt_file": self.prompt_file,
                "prompt_length_chars": len(self.prompt_text),
                "error": str(e),
                "processing_time_sec": 0.0,
                "success": False
            })
            tqdm.write(f"❌ Error processing {image_path}: {e}")

    def process_folder(self):
        imgs = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img in tqdm(imgs, desc="Processing BPMN Images with Qwen2.5-VL"):
            self.process_single_image(os.path.join(self.image_folder, img))

    def calculate_metrics(self):
        total = self.total_prompt_tokens + self.total_output_tokens
        m = {
            "model_used": self.model_id,
            "prompt_file": self.prompt_file,
            "prompt_text": self.prompt_text,
            "total_images_processed": self.total_images_processed,
            "total_prompt_tokens": int(self.total_prompt_tokens),
            "total_completion_tokens": int(self.total_output_tokens),
            "overall_total_tokens": int(total),
            "total_processing_time_seconds": self.total_processing_time,
        }
        if self.total_images_processed:
            m["avg_prompt_tokens_per_image"] = self.total_prompt_tokens / self.total_images_processed
            m["avg_completion_tokens_per_image"] = self.total_output_tokens / self.total_images_processed
            m["avg_total_tokens_per_image"] = total / self.total_images_processed
            m["avg_processing_time_per_image_seconds"] = self.total_processing_time / self.total_images_processed
        else:
            m["avg_prompt_tokens_per_image"] = 0
            m["avg_completion_tokens_per_image"] = 0
            m["avg_total_tokens_per_image"] = 0
            m["avg_processing_time_per_image_seconds"] = 0
        return m

    def save_detailed_logs(self, log_file="qwen_token_log.json"):
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.detailed_logs, f, indent=4)
        print(f"Detailed logs saved to {log_file}")
