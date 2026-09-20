"""
Generate simulated JEPA vs Generative AI comparison data.

Based on real research findings:
- I-JEPA (Meta, 2023) uses ~60-75% less compute than generative vision models
- V-JEPA uses ~70-80% less compute than pixel-prediction models
- Sources: Meta AI Research papers on I-JEPA and V-JEPA
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

GRID_INTENSITY = 0.708  # kg CO2 per kWh (India)

# Tasks that both architectures can perform
VISION_TASKS = [
    "Image Classification (ImageNet)",
    "Object Detection (COCO)",
    "Semantic Segmentation",
    "Image Feature Extraction",
    "Scene Understanding",
    "Action Recognition (Video)",
    "Depth Estimation",
    "Image Similarity Search",
    "Visual Question Answering",
    "Image Retrieval",
]

def generate_jepa_comparison():
    """Compare JEPA (Predictive) vs Generative vision models on energy."""
    rows = []
    query_id = 0

    models = [
        {
            "name": "ViT-MAE (Generative)",
            "type": "Generative",
            "params_b": 0.3,
            "base_energy_j": 850,     # Joules per inference batch
            "base_latency": 2.8,      # seconds
            "base_memory": 4200,      # MB
        },
        {
            "name": "Stable Diffusion (Generative)",
            "type": "Generative",
            "params_b": 0.9,
            "base_energy_j": 2400,
            "base_latency": 8.5,
            "base_memory": 7800,
        },
        {
            "name": "I-JEPA (Predictive)",
            "type": "Predictive (JEPA)",
            "params_b": 0.3,
            "base_energy_j": 280,     # ~67% less than ViT-MAE
            "base_latency": 0.9,
            "base_memory": 2800,
        },
        {
            "name": "V-JEPA (Predictive)",
            "type": "Predictive (JEPA)",
            "params_b": 0.6,
            "base_energy_j": 520,     # ~78% less than Stable Diffusion
            "base_latency": 1.8,
            "base_memory": 3500,
        },
    ]

    for model in models:
        for task in VISION_TASKS:
            for run in range(5):  # 5 runs per task
                noise = np.random.normal(1.0, 0.08)
                energy_j = max(10, model["base_energy_j"] * noise)
                energy_kwh = energy_j / 3_600_000
                latency = max(0.1, model["base_latency"] * noise)
                memory = max(500, model["base_memory"] + np.random.normal(0, 100))
                co2_grams = energy_kwh * GRID_INTENSITY * 1000

                rows.append({
                    "query_id": query_id,
                    "model": model["name"],
                    "architecture_type": model["type"],
                    "parameters_b": model["params_b"],
                    "task": task,
                    "energy_j": round(energy_j, 2),
                    "energy_kwh": round(energy_kwh, 8),
                    "duration_s": round(latency, 3),
                    "co2_grams": round(co2_grams, 6),
                    "gpu_memory_mb": round(memory, 1),
                })
                query_id += 1

    return pd.DataFrame(rows)


def generate_jepa_scaling():
    """Show how JEPA scales more efficiently with model size."""
    rows = []
    # Generative models: energy scales ~quadratically with params
    # JEPA models: energy scales ~linearly with params
    param_sizes = [0.1, 0.3, 0.6, 1.0, 2.0, 5.0]

    for params in param_sizes:
        for run in range(10):
            noise_gen = np.random.normal(1.0, 0.1)
            noise_jepa = np.random.normal(1.0, 0.1)

            # Generative: quadratic scaling (params^1.8)
            gen_energy_j = 300 * (params ** 1.8) * noise_gen
            gen_kwh = gen_energy_j / 3_600_000

            # JEPA: linear scaling (params^1.1)
            jepa_energy_j = 300 * (params ** 1.1) * noise_jepa
            jepa_kwh = jepa_energy_j / 3_600_000

            rows.append({
                "parameters_b": params,
                "architecture": "Generative",
                "energy_j": round(gen_energy_j, 2),
                "energy_kwh": round(gen_kwh, 8),
                "co2_grams": round(gen_kwh * GRID_INTENSITY * 1000, 6),
            })
            rows.append({
                "parameters_b": params,
                "architecture": "JEPA (Predictive)",
                "energy_j": round(jepa_energy_j, 2),
                "energy_kwh": round(jepa_kwh, 8),
                "co2_grams": round(jepa_kwh * GRID_INTENSITY * 1000, 6),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating JEPA comparison data...")
    comp_df = generate_jepa_comparison()
    comp_df.to_csv(os.path.join(data_dir, "jepa_comparison.csv"), index=False)
    print(f"  -> {len(comp_df)} rows")

    print("Generating JEPA scaling data...")
    scale_df = generate_jepa_scaling()
    scale_df.to_csv(os.path.join(data_dir, "jepa_scaling.csv"), index=False)
    print(f"  -> {len(scale_df)} rows")

    print("\n✅ JEPA data generated!")
