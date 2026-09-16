"""
Generate realistic sample data for the AI Inference Emissions Dashboard.

The numbers are grounded in published benchmarks and real-world measurements
from consumer GPUs (NVIDIA RTX 3090 / 4090 class hardware). Each model's
energy profile is seeded around plausible per-inference values so the
dashboard looks realistic even before real CodeCarbon data is collected.

Run:  python data/generate_data.py
"""

import os
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────

# India average grid carbon intensity (kg CO₂ / kWh)
GRID_INTENSITY = 0.708

QUERIES_PER_MODEL = 50  # per model, per precision level

PROMPTS = [
    "What is photosynthesis?",
    "Explain Newton's third law of motion.",
    "Summarize the plot of Romeo and Juliet.",
    "What causes earthquakes?",
    "Write a short poem about the ocean.",
    "Explain how vaccines work in simple terms.",
    "What is the difference between weather and climate?",
    "Describe the water cycle.",
    "Who was Ada Lovelace and why is she important?",
    "What are the main causes of air pollution?",
    "Explain the greenhouse effect.",
    "What is machine learning in one paragraph?",
    "Describe how a battery works.",
    "What are renewable energy sources?",
    "Explain why the sky is blue.",
    "What is the theory of evolution?",
    "Describe the structure of DNA.",
    "How does the internet work?",
    "What is inflation in economics?",
    "Explain the concept of supply and demand.",
]

# Model profiles: (name, params_b, avg_energy_kwh, energy_std, avg_duration_s, duration_std, avg_tokens, gpu_memory_mb_mean)
MODELS = [
    ("TinyLlama-1.1B", 1.1,  0.00018, 0.00005, 1.2,  0.3,  85,  2200),
    ("Phi-2-2.7B",     2.7,  0.00042, 0.00010, 2.1,  0.5, 110,  5400),
    ("Mistral-7B",     7.2,  0.00120, 0.00025, 4.8,  1.0, 130, 14400),
    ("Llama-2-13B",    13.0, 0.00260, 0.00050, 8.5,  1.8, 145, 26000),
]

# Quantization profiles for Mistral-7B (precision, energy_multiplier, quality_score_mean, quality_std, memory_mb_mean)
QUANT_PROFILES = [
    ("FP16", 1.00, 96.0, 1.5, 14400),
    ("INT8", 0.65, 93.5, 2.0, 7200),
    ("INT4", 0.42, 87.0, 3.0, 3600),
]


def generate_inference_results() -> pd.DataFrame:
    """Generate per-query inference measurement data."""
    rows = []
    query_id = 0

    for model_name, params_b, avg_energy, energy_std, avg_dur, dur_std, avg_tokens, mem_mean in MODELS:
        for i in range(QUERIES_PER_MODEL):
            prompt = PROMPTS[i % len(PROMPTS)]
            energy = max(0.00001, np.random.normal(avg_energy, energy_std))
            duration = max(0.1, np.random.normal(avg_dur, dur_std))
            tokens = max(20, int(np.random.normal(avg_tokens, 15)))
            co2_grams = energy * GRID_INTENSITY * 1000  # kWh → g CO₂
            energy_per_token = energy / tokens
            gpu_memory = max(100, int(np.random.normal(mem_mean, 500)))

            rows.append({
                "query_id": query_id,
                "model": model_name,
                "parameters_b": params_b,
                "precision": "FP16",
                "prompt": prompt,
                "tokens_generated": tokens,
                "energy_kwh": round(energy, 8),
                "duration_s": round(duration, 3),
                "co2_grams": round(co2_grams, 6),
                "energy_per_token": round(energy_per_token, 8),
                "gpu_memory_mb": gpu_memory
            })
            query_id += 1

    return pd.DataFrame(rows)


def generate_quantization_data() -> pd.DataFrame:
    """Generate Mistral-7B quantization comparison data."""
    rows = []
    query_id = 0

    # Mistral-7B baseline values
    base_energy = 0.00120
    base_std = 0.00025
    base_dur = 4.8
    base_dur_std = 1.0
    base_tokens = 130

    for precision, energy_mult, quality_mean, quality_std, mem_mean in QUANT_PROFILES:
        for i in range(QUERIES_PER_MODEL):
            prompt = PROMPTS[i % len(PROMPTS)]
            energy = max(0.00001, np.random.normal(base_energy * energy_mult, base_std * energy_mult))
            duration = max(0.1, np.random.normal(base_dur * energy_mult, base_dur_std * energy_mult))
            tokens = max(20, int(np.random.normal(base_tokens, 15)))
            co2_grams = energy * GRID_INTENSITY * 1000
            quality = min(100.0, max(50.0, np.random.normal(quality_mean, quality_std)))
            energy_per_token = energy / tokens
            gpu_memory = max(100, int(np.random.normal(mem_mean, 200)))

            rows.append({
                "query_id": query_id,
                "model": "Mistral-7B",
                "parameters_b": 7.2,
                "precision": precision,
                "prompt": prompt,
                "tokens_generated": tokens,
                "energy_kwh": round(energy, 8),
                "duration_s": round(duration, 3),
                "co2_grams": round(co2_grams, 6),
                "quality_score": round(quality, 2),
                "energy_per_token": round(energy_per_token, 8),
                "gpu_memory_mb": gpu_memory
            })
            query_id += 1

    return pd.DataFrame(rows)


def generate_batch_size_data() -> pd.DataFrame:
    """Generate Mistral-7B data for different batch sizes."""
    rows = []
    query_id = 0
    batch_sizes = [1, 2, 4]
    base_energy = 0.00120
    base_dur = 4.8
    base_tokens = 130
    mem_mean = 14400

    for bs in batch_sizes:
        if bs == 1:
            e_mult = 1.0
            d_mult = 1.0
        elif bs == 2:
            e_mult = 0.55
            d_mult = 0.70
        elif bs == 4:
            e_mult = 0.3025 # 0.55 * 0.55
            d_mult = 0.49   # 0.70 * 0.70

        for prompt in PROMPTS:
            energy = max(0.00001, np.random.normal(base_energy * e_mult, base_energy * 0.2 * e_mult))
            duration = max(0.1, np.random.normal(base_dur * d_mult, base_dur * 0.2 * d_mult))
            tokens = max(20, int(np.random.normal(base_tokens, 15)))
            co2_grams = energy * GRID_INTENSITY * 1000
            energy_per_token = energy / tokens
            gpu_memory = max(100, int(np.random.normal(mem_mean, 500)))

            rows.append({
                "query_id": query_id,
                "model": "Mistral-7B",
                "batch_size": bs,
                "prompt": prompt,
                "tokens_generated": tokens,
                "energy_kwh": round(energy, 8),
                "duration_s": round(duration, 3),
                "co2_grams": round(co2_grams, 6),
                "energy_per_token": round(energy_per_token, 8),
                "gpu_memory_mb": gpu_memory
            })
            query_id += 1
            
    return pd.DataFrame(rows)


def generate_input_length_data() -> pd.DataFrame:
    """Generate Mistral-7B data for different input lengths."""
    rows = []
    query_id = 0
    categories = [
        ("Short", 15, 1.0, 1.0),
        ("Medium", 60, 0.3, 1.5),
        ("Long", 150, 0.18, 2.1)
    ]
    base_tokens = 130

    for cat_name, in_tokens, e_mult, e_tot_mult in categories:
        for run in range(3):
            for prompt in PROMPTS[:5]:
                tokens_in = int(np.random.normal(in_tokens, in_tokens * 0.1))
                tokens = max(20, int(np.random.normal(base_tokens, 15)))
                
                energy = max(0.00001, np.random.normal(0.00120 * e_tot_mult, 0.00120 * 0.1 * e_tot_mult))
                energy_per_token = energy / (tokens + tokens_in) # Simple approximation of energy per token
                duration = max(0.1, np.random.normal(4.8 * e_tot_mult, 4.8 * 0.2 * e_tot_mult))
                co2_grams = energy * GRID_INTENSITY * 1000

                rows.append({
                    "query_id": query_id,
                    "model": "Mistral-7B",
                    "input_length_category": cat_name,
                    "input_tokens": tokens_in,
                    "tokens_generated": tokens,
                    "energy_kwh": round(energy, 8),
                    "duration_s": round(duration, 3),
                    "co2_grams": round(co2_grams, 6),
                    "energy_per_token": round(energy_per_token, 8)
                })
                query_id += 1

    return pd.DataFrame(rows)


def generate_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate inference results into per-model summary stats."""
    summary = df.groupby("model").agg(
        parameters_b=("parameters_b", "first"),
        total_queries=("query_id", "count"),
        avg_energy_kwh=("energy_kwh", "mean"),
        std_energy_kwh=("energy_kwh", "std"),
        avg_co2_grams=("co2_grams", "mean"),
        std_co2_grams=("co2_grams", "std"),
        avg_duration_s=("duration_s", "mean"),
        avg_tokens=("tokens_generated", "mean"),
        avg_energy_per_token=("energy_per_token", "mean"),
        avg_gpu_memory_mb=("gpu_memory_mb", "mean")
    ).reset_index()

    # Round for readability
    for col in ["avg_energy_kwh", "std_energy_kwh", "avg_energy_per_token"]:
        summary[col] = summary[col].round(8)
    for col in ["avg_co2_grams", "std_co2_grams"]:
        summary[col] = summary[col].round(6)
    summary["avg_duration_s"] = summary["avg_duration_s"].round(3)
    summary["avg_tokens"] = summary["avg_tokens"].round(1)
    summary["avg_gpu_memory_mb"] = summary["avg_gpu_memory_mb"].round(1)

    return summary


def main():
    """Generate all CSV files."""
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Per-query inference results
    print("Generating inference_results.csv ...")
    inference_df = generate_inference_results()
    inference_df.to_csv(os.path.join(data_dir, "inference_results.csv"), index=False)
    print(f"  → {len(inference_df)} rows")

    # 2. Model summary
    print("Generating model_summary.csv ...")
    summary_df = generate_model_summary(inference_df)
    summary_df.to_csv(os.path.join(data_dir, "model_summary.csv"), index=False)
    print(f"  → {len(summary_df)} rows")

    # 3. Quantization comparison
    print("Generating quantization_comparison.csv ...")
    quant_df = generate_quantization_data()
    quant_df.to_csv(os.path.join(data_dir, "quantization_comparison.csv"), index=False)
    print(f"  → {len(quant_df)} rows")
    
    # 4. Batch size comparison
    print("Generating batch_size_results.csv ...")
    batch_size_df = generate_batch_size_data()
    batch_size_df.to_csv(os.path.join(data_dir, "batch_size_results.csv"), index=False)
    print(f"  → {len(batch_size_df)} rows")

    # 5. Input length comparison
    print("Generating input_length_results.csv ...")
    input_length_df = generate_input_length_data()
    input_length_df.to_csv(os.path.join(data_dir, "input_length_results.csv"), index=False)
    print(f"  → {len(input_length_df)} rows")


    print("\n✅ All data files generated successfully.")


if __name__ == "__main__":
    main()
