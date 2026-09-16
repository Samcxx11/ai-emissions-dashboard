import time
import torch
from codecarbon import EmissionsTracker
from benchmark.config import CODECARBON_OUTPUT_DIR

def measure_single_inference(model, tokenizer, prompt, max_new_tokens, device):
    tracker = EmissionsTracker(
        save_to_file=False,
        log_level="error",
        output_dir=CODECARBON_OUTPUT_DIR
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens_input = inputs.input_ids.shape[1]
    
    tracker.start()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    end_time = time.time()
    emissions_data = tracker.stop()
    
    duration_s = end_time - start_time
    
    # Calculate tokens generated (excluding input prompt tokens if included in output)
    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
    
    if emissions_data is None:
        energy_kwh = 0.0
        co2_grams = 0.0
    else:
        # Use final emissions data if available, otherwise try to extract from tracker
        try:
             energy_kwh = tracker.final_emissions_data.energy_consumed
             co2_grams = tracker.final_emissions_data.emissions * 1000
        except AttributeError:
             energy_kwh = 0.0
             co2_grams = 0.0
             
    # Fallback if CodeCarbon fails to get energy
    if energy_kwh == 0.0:
        # Estimate using TDP (assuming 250W average GPU power for estimation if no real data)
        power_w = 250.0 
        energy_kwh = (power_w * duration_s) / (3600 * 1000)
        co2_grams = energy_kwh * 708 # Using grid carbon intensity

    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_max_memory_allocated()
    else:
        gpu_memory_mb = 0.0
        
    return {
        "energy_kwh": energy_kwh,
        "co2_grams": co2_grams,
        "duration_s": duration_s,
        "tokens_generated": tokens_generated,
        "tokens_input": tokens_input,
        "gpu_memory_mb": gpu_memory_mb
    }

def measure_batch_inference(model, tokenizer, prompts, max_new_tokens, device):
    tracker = EmissionsTracker(
        save_to_file=False,
        log_level="error",
        output_dir=CODECARBON_OUTPUT_DIR
    )
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    batch_size = len(prompts)
    
    tracker.start()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    end_time = time.time()
    emissions_data = tracker.stop()
    
    duration_s = end_time - start_time
    
    if emissions_data is None:
        energy_kwh = 0.0
        co2_grams = 0.0
    else:
        try:
             energy_kwh = tracker.final_emissions_data.energy_consumed
             co2_grams = tracker.final_emissions_data.emissions * 1000
        except AttributeError:
             energy_kwh = 0.0
             co2_grams = 0.0
             
    if energy_kwh == 0.0:
        power_w = 250.0 
        energy_kwh = (power_w * duration_s) / (3600 * 1000)
        co2_grams = energy_kwh * 708

    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_max_memory_allocated()
    else:
        gpu_memory_mb = 0.0
        
    results = []
    # Distribute energy equally among batch items
    energy_per_item = energy_kwh / batch_size
    co2_per_item = co2_grams / batch_size
    duration_per_item = duration_s / batch_size
    
    for i in range(batch_size):
        tokens_input_i = inputs.attention_mask[i].sum().item()
        # Count actual generated tokens (excluding padding in output)
        out_tokens_i = outputs[i][inputs.input_ids.shape[1]:]
        tokens_generated_i = (out_tokens_i != tokenizer.pad_token_id).sum().item()
        
        results.append({
            "energy_kwh": energy_per_item,
            "co2_grams": co2_per_item,
            "duration_s": duration_per_item,
            "tokens_generated": tokens_generated_i,
            "tokens_input": tokens_input_i,
            "gpu_memory_mb": gpu_memory_mb / batch_size
        })
        
    return results

def calculate_energy_per_token(energy_kwh, tokens_generated):
    if tokens_generated > 0:
        return energy_kwh / tokens_generated
    return 0.0
