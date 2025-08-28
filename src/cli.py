
import os, argparse, json
from src.pipelines.gpt4o_redrawer import GPT4oRedrawerPipeline
from src.pipelines.mistral_redrawer import MistralBPMNRedrawerPipeline
from src.pipelines.gemini_redrawer import GeminiRedrawerPipeline
from src.pipelines.gemma_redrawer import GemmaRedrawerPipeline
from src.pipelines.qwen_redrawer import QwenVLRedrawerPipeline
from src.utils import save_metrics, save_metrics_csv

def main():
    p = argparse.ArgumentParser(description="BPMN diagram image → BPMN XML (multi-model suite)")
    p.add_argument("--engine", required=True, choices=["openai","mistral","gemini","gemma","qwen"])
    p.add_argument("--image-folder", required=True, help="Folder with input images (.png/.jpg/.jpeg)")
    p.add_argument("--prompt-file", required=True, help="Prompt text file")
    p.add_argument("--out-dir", default="outputs", help="Output directory")
    p.add_argument("--model", help="Model name / HF ID as appropriate")
    p.add_argument("--api-key", help="API key (OpenAI/Mistral/Gemini). Defaults to corresponding env var.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.engine == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Missing OpenAI API key. Use --api-key or set OPENAI_API_KEY.")
        model = args.model or "gpt-4o"
        pipe = GPT4oRedrawerPipeline(api_key=api_key, prompt_file=args.prompt_file, model=model, output_folder=args.out_dir)
        pipe.process_folder(args.image_folder)
        metrics = pipe.calculate_metrics()
        save_metrics(metrics, os.path.join(args.out_dir, "metrics_openai.json"))
        save_metrics_csv(metrics, os.path.join(args.out_dir, "metrics_openai.csv"))
        pipe.save_detailed_logs(os.path.join(args.out_dir, "logs_openai.json"))
        return

    if args.engine == "mistral":
        api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise SystemExit("Missing Mistral API key. Use --api-key or set MISTRAL_API_KEY.")
        model = args.model or "mistral-small-2503"
        pipe = MistralBPMNRedrawerPipeline(api_key=api_key, prompt_file=args.prompt_file, output_folder=args.out_dir, model=model)
        pipe.process_folder(args.image_folder)
        metrics = pipe.calculate_metrics()
        save_metrics(metrics, os.path.join(args.out_dir, "metrics_mistral.json"))
        save_metrics_csv(metrics, os.path.join(args.out_dir, "metrics_mistral.csv"))
        pipe.save_detailed_logs(os.path.join(args.out_dir, "logs_mistral.json"))
        return

    if args.engine == "gemini":
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Missing Google API key. Use --api-key or set GOOGLE_API_KEY.")
        model = args.model or "gemini-2.5-flash"
        pipe = GeminiRedrawerPipeline(api_key=api_key, prompt_file=args.prompt_file, image_folder=args.image_folder, output_folder=args.out_dir, model=model)
        pipe.process_folder()
        metrics = pipe.calculate_metrics()
        save_metrics(metrics, os.path.join(args.out_dir, "metrics_gemini.json"))
        save_metrics_csv(metrics, os.path.join(args.out_dir, "metrics_gemini.csv"))
        pipe.save_detailed_logs(os.path.join(args.out_dir, "logs_gemini.json"))
        return

    if args.engine == "gemma":
        model_id = args.model or "google/gemma-3-4b-it"
        pipe = GemmaRedrawerPipeline(model_id=model_id, prompt_file=args.prompt_file, image_folder=args.image_folder, output_folder=args.out_dir)
        pipe.process_folder()
        metrics = pipe.calculate_metrics()
        save_metrics(metrics, os.path.join(args.out_dir, "metrics_gemma.json"))
        save_metrics_csv(metrics, os.path.join(args.out_dir, "metrics_gemma.csv"))
        pipe.save_detailed_logs(os.path.join(args.out_dir, "logs_gemma.json"))
        return

    if args.engine == "qwen":
        model_id = args.model or "Qwen/Qwen2.5-VL-3B-Instruct"
        pipe = QwenVLRedrawerPipeline(model_id=model_id, prompt_file=args.prompt_file, image_folder=args.image_folder, output_folder=args.out_dir)
        pipe.process_folder()
        metrics = pipe.calculate_metrics()
        save_metrics(metrics, os.path.join(args.out_dir, "metrics_qwen.json"))
        save_metrics_csv(metrics, os.path.join(args.out_dir, "metrics_qwen.csv"))
        pipe.save_detailed_logs(os.path.join(args.out_dir, "logs_qwen.json"))
        return

if __name__ == "__main__":
    main()
