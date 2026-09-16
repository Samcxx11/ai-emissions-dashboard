import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from benchmark.config import MODELS

def load_model(model_name: str, precision: str = "FP16"):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in config.MODELS.")
    
    hf_model_id = MODELS[model_name]
    print(f"Loading {hf_model_id} with precision {precision}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        kwargs = {"device_map": "auto"}
        
        if precision == "FP16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "INT8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "INT4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported precision: {precision}")
            
        model = AutoModelForCausalLM.from_pretrained(hf_model_id, **kwargs)
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model {model_name} ({hf_model_id}): {e}")
        raise

def cleanup_model(model, tokenizer):
    print("Cleaning up model and tokenizer...")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_max_memory_allocated()
        return mem
    return 0.0
