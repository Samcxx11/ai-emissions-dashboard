import argparse
import os
import pandas as pd
from tqdm import tqdm

from benchmark.config import (
    MODELS, STANDARD_PROMPTS, SHORT_PROMPTS, MEDIUM_PROMPTS, LONG_PROMPTS,
    QUANTIZATION_LEVELS, BATCH_SIZES, MAX_NEW_TOKENS, DATA_DIR
)
from benchmark.models import load_model, cleanup_model, get_device
from benchmark.measure import measure_single_inference, measure_batch_inference, calculate_energy_per_token

def save_incremental(df_new, filepath):
    if not os.path.exists(filepath):
        df_new.to_csv(filepath, index=False)
    else:
        df_new.to_csv(filepath, mode='a', header=False, index=False)

def get_params_b(model_name):
    if "1.1B" in model_name: return 1.1
    if "2.7B" in model_name: return 2.7
    if "7B" in model_name: return 7.2
    if "13B" in model_name: return 13.0
    return 0.0

def run_model_comparison(models_to_test, prompts, output_dir):
    print(f"Running Model Comparison for {len(models_to_test)} models...")
    device = get_device()
    output_file = os.path.join(output_dir, "inference_results.csv")
    
    all_results = []
    
    for model_name in models_to_test:
        try:
            model, tokenizer = load_model(model_name, precision="FP16")
            model.eval()
            
            for i, prompt in enumerate(tqdm(prompts, desc=f"Testing {model_name}")):
                res = measure_single_inference(model, tokenizer, prompt, MAX_NEW_TOKENS, device)
                
                row = {
                    "query_id": i,
                    "model": model_name,
                    "parameters_b": get_params_b(model_name),
                    "precision": "FP16",
                    "prompt": prompt,
                    "tokens_generated": res["tokens_generated"],
                    "tokens_input": res["tokens_input"],
                    "energy_kwh": res["energy_kwh"],
                    "duration_s": res["duration_s"],
                    "co2_grams": res["co2_grams"],
                    "energy_per_token": calculate_energy_per_token(res["energy_kwh"], res["tokens_generated"]),
                    "gpu_memory_mb": res["gpu_memory_mb"]
                }
                all_results.append(row)
                save_incremental(pd.DataFrame([row]), output_file)
                
            cleanup_model(model, tokenizer)
        except Exception as e:
            print(f"Skipping model {model_name} due to error: {e}")
            
    if all_results:
        df = pd.DataFrame(all_results)
        summary = df.groupby("model").agg({
            "energy_kwh": "mean",
            "duration_s": "mean",
            "tokens_generated": "mean",
            "energy_per_token": "mean",
            "gpu_memory_mb": "max"
        }).reset_index()
        summary.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)

def run_quantization_sweep(prompts, output_dir):
    print("Running Quantization Sweep on Mistral-7B...")
    device = get_device()
    output_file = os.path.join(output_dir, "quantization_comparison.csv")
    model_name = "Mistral-7B"
    
    for precision in QUANTIZATION_LEVELS:
        try:
            model, tokenizer = load_model(model_name, precision=precision)
            model.eval()
            
            for i, prompt in enumerate(tqdm(prompts, desc=f"Testing {precision}")):
                res = measure_single_inference(model, tokenizer, prompt, MAX_NEW_TOKENS, device)
                
                row = {
                    "query_id": i,
                    "model": model_name,
                    "parameters_b": get_params_b(model_name),
                    "precision": precision,
                    "prompt": prompt,
                    "tokens_generated": res["tokens_generated"],
                    "tokens_input": res["tokens_input"],
                    "energy_kwh": res["energy_kwh"],
                    "duration_s": res["duration_s"],
                    "co2_grams": res["co2_grams"],
                    "energy_per_token": calculate_energy_per_token(res["energy_kwh"], res["tokens_generated"]),
                    "gpu_memory_mb": res["gpu_memory_mb"],
                    "quality_score": 0.0 # Placeholder
                }
                save_incremental(pd.DataFrame([row]), output_file)
                
            cleanup_model(model, tokenizer)
        except Exception as e:
            print(f"Skipping precision {precision} due to error: {e}")

def run_batch_size_sweep(prompts, output_dir):
    print("Running Batch Size Sweep on Mistral-7B...")
    device = get_device()
    output_file = os.path.join(output_dir, "batch_size_results.csv")
    model_name = "Mistral-7B"
    
    try:
        model, tokenizer = load_model(model_name, precision="FP16")
        model.eval()
        
        for batch_size in BATCH_SIZES:
            # Create batches from prompts
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Batch size {batch_size}"):
                batch_prompts = prompts[i:i+batch_size]
                if not batch_prompts: continue
                
                results = measure_batch_inference(model, tokenizer, batch_prompts, MAX_NEW_TOKENS, device)
                
                rows = []
                for j, res in enumerate(results):
                    row = {
                        "query_id": i + j,
                        "model": model_name,
                        "batch_size": batch_size,
                        "prompt": batch_prompts[j],
                        "tokens_generated": res["tokens_generated"],
                        "energy_kwh": res["energy_kwh"],
                        "duration_s": res["duration_s"],
                        "co2_grams": res["co2_grams"],
                        "energy_per_token": calculate_energy_per_token(res["energy_kwh"], res["tokens_generated"]),
                        "gpu_memory_mb": res["gpu_memory_mb"]
                    }
                    rows.append(row)
                
                save_incremental(pd.DataFrame(rows), output_file)
                
        cleanup_model(model, tokenizer)
    except Exception as e:
        print(f"Failed to run batch size sweep: {e}")

def run_input_length_sweep(output_dir):
    print("Running Input Length Sweep on Mistral-7B...")
    device = get_device()
    output_file = os.path.join(output_dir, "input_length_results.csv")
    model_name = "Mistral-7B"
    
    categories = {
        "short": SHORT_PROMPTS,
        "medium": MEDIUM_PROMPTS,
        "long": LONG_PROMPTS
    }
    
    try:
        model, tokenizer = load_model(model_name, precision="FP16")
        model.eval()
        
        query_id = 0
        for category, prompts in categories.items():
            for prompt in tqdm(prompts, desc=f"Testing {category} length"):
                res = measure_single_inference(model, tokenizer, prompt, MAX_NEW_TOKENS, device)
                
                row = {
                    "query_id": query_id,
                    "model": model_name,
                    "input_length_category": category,
                    "input_tokens": res["tokens_input"],
                    "tokens_generated": res["tokens_generated"],
                    "energy_kwh": res["energy_kwh"],
                    "duration_s": res["duration_s"],
                    "co2_grams": res["co2_grams"],
                    "energy_per_token": calculate_energy_per_token(res["energy_kwh"], res["tokens_generated"])
                }
                save_incremental(pd.DataFrame([row]), output_file)
                query_id += 1
                
        cleanup_model(model, tokenizer)
    except Exception as e:
        print(f"Failed to run input length sweep: {e}")

def main():
    parser = argparse.ArgumentParser(description="LLM Inference Energy Benchmarking")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "model_comparison", "quantization", "batch_size", "input_length"],
                        help="Which experiment to run")
    parser.add_argument("--models", type=str, nargs="+", default=list(MODELS.keys()),
                        help="List of models to benchmark")
    parser.add_argument("--num-prompts", type=int, default=20,
                        help="Number of standard prompts to use")
    parser.add_argument("--output-dir", type=str, default=DATA_DIR,
                        help="Directory to save CSV results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    
    args = parser.parse_args()
    
    # Filter models
    models_to_test = [m for m in args.models if m in MODELS]
    if not models_to_test:
        print("No valid models selected. Available:", list(MODELS.keys()))
        return
        
    prompts = STANDARD_PROMPTS[:args.num_prompts]
    
    if args.dry_run:
        print("--- DRY RUN PLAN ---")
        print(f"Experiment: {args.experiment}")
        print(f"Models: {models_to_test}")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Output directory: {args.output_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "codecarbon"), exist_ok=True)
    
    if args.experiment in ["all", "model_comparison"]:
        run_model_comparison(models_to_test, prompts, args.output_dir)
        
    if args.experiment in ["all", "quantization"]:
        run_quantization_sweep(prompts, args.output_dir)
        
    if args.experiment in ["all", "batch_size"]:
        run_batch_size_sweep(prompts, args.output_dir)
        
    if args.experiment in ["all", "input_length"]:
        run_input_length_sweep(args.output_dir)
        
    print("Benchmarking completed!")

if __name__ == "__main__":
    main()
