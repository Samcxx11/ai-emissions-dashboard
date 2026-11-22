# AI Inference Emissions Dashboard & Research Telemetry Engine

An empirical research platform, interactive telemetry engine, and mitigation framework for evaluating the electrical energy consumption (Watt-hours) and greenhouse gas emissions (gCO2e) of Large Language Models (LLMs) and Small Language Models (SLMs) during operational inference.

Accompanied by the comprehensive academic submission report: [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

---

## Executive Overview

While public discourse emphasizes the carbon footprint of AI model pre-training, empirical telemetry reveals that for deployed commercial systems, **operational inference accounts for 80% to 90% of total lifecycle energy consumption**.

This project provides a comprehensive experimental evaluation across open-source autoregressive transformer architectures (1.1B to 13.0B parameters), examining the impact of parameter scale, post-training quantization, request batch concurrency, prompt length, and architectural paradigms such as Yann LeCun's **Joint-Embedding Predictive Architecture (JEPA)** and inference optimizations (**Automatic Prefix Caching, SLO-Aware GPU DVFS, and Sliding Window Attention**).

The dashboard includes **live host hardware telemetry** and **on-device Apple Silicon Metal Performance Shaders (MPS) benchmarking** calibrated to local laptop hardware.

---

## Key Empirical Findings

1. **Super-Linear Scaling Penalty (14.5x Multiplier):**  
   Scaling model capacity from 1.1B (TinyLlama) to 13.0B (Llama-2) yields a **1,353% increase in per-query energy consumption (0.184 Wh vs. 2.681 Wh)**, demonstrating that right-sizing model capacity to task complexity is the single most potent operational lever for carbon abatement.

2. **The INT8 Quantization Sweet Spot:**  
   Post-training quantization from FP16 to INT8 on Llama-2-7B achieves an immediate **35.0% reduction in electricity and greenhouse gas emissions** with a negligible **1.6% degradation in semantic quality score (94.9 vs. 96.5)**. INT4 yields a 58.1% energy reduction with a 7.2% quality penalty.

3. **Batch Scheduling Power Amortization:**  
   Increasing batch concurrency from $N=1$ to $N=4$ amortizes static GPU idle baseline power, achieving a **21.4% per-query energy reduction**.

4. **Input Context Prefill Overhead:**  
   Prompt token ingestion incurs quadratic attention FLOPs ($O(N^2)$ prefill complexity). Pruning redundant conversational framing and RAG context chunks reduces prompt ingestion energy by 10% to 25%.

5. **Compounded Multiplicative Mitigation Law:**  
   Across a 22-strategy mitigation taxonomy, combining architectural right-sizing, INT8 quantization, semantic caching, dynamic batching, cascading routing, and renewable grid arbitrage achieves **over 98.2% cumulative carbon abatement**.

6. **On-Device Apple M4 Metal Benchmarking:**  
   Direct tensor execution on the local Apple M4 GPU via PyTorch MPS reveals that **Automatic Prefix Caching (APC) delivers a 13.4x to 37.9x prefill speedup (92.5% to 97.4% energy reduction)** by reusing precomputed KV-cache blocks, while **Sliding Window Attention (SWA) achieves a 10.9x to 21.4x speedup (90.8% to 95.3% energy reduction)** by bounding attention to local context.

---

## Chapter Architecture

The dashboard is structured into 10 cohesive research chapters:

| Chapter | Module | Core Research Focus |
| :--- | :--- | :--- |
| **00** | `app.py` | **Executive Summary & Scope:** Empirical scope, hardware register telemetry, live host status badge, methodology, peer-reviewed foundations. |
| **01** | `1_Energy_CO2_Comparison.py` | **Architecture Scale:** Parameter scaling dynamics (1.1B to 13B), Wh vs. gCO2e trade-offs, super-linear regression. |
| **02** | `2_Quantization_Impact.py` | **Precision Dynamics:** FP16 vs. INT8 vs. INT4 post-training quantization, energy abatement vs. quality retention curves. |
| **03** | `3_Per_Query_Analysis.py` | **Statistical Dispersion:** 200 standardized runs, energy histograms, interquartile box plots, prompt-level telemetry ledger. |
| **04** | `4_Real_World_Impact.py` | **Macro-Scale Production Calculator:** Extrapolating daily query volumes (1K to 10M) to annual metric tons, EPA vehicle miles, and tree sequestration equivalents. |
| **05** | `5_Batch_Size_Analysis.py` | **Batch Scheduling Dynamics:** Tensor core saturation, power amortization across batch sizes $N=1, 2, 4$, throughput gains. |
| **06** | `6_Input_Length_Impact.py` | **Prompt Length Scaling:** Quadratic prefill ($O(N^2)$) vs. linear autoregressive decode ($O(N)$), context compression savings. |
| **07** | `7_Carbon_Reduction_Strategies.py` | **Mitigation Framework:** 22 actionable engineering strategies across 4 tiers, interactive compounding waterfall abatement simulator. |
| **08** | `8_JEPA_Carbon_Reduction.py` | **JEPA Multimodal Framework:** Latent prediction objective ($\hat{y} = P(E_c(x_c))$) vs. pixel reconstruction, 5-stage multimodal compression pipeline, energy budget formulations. |
| **09** | `9_Inference_Optimization_Methods.py` | **Top 3 Inference Optimizations:** Automatic Prefix Caching, SLO-Aware GPU DVFS, Sliding Window Attention, live M4 hardware telemetry, and **real-time on-device Metal benchmark runner**. |

---

## Standardized Carbon Accounting

Operational emissions are calculated following standardized international reporting guidelines:

$$\text{Operational CO}_2\text{e} = E_{\text{consumed}} \;(\text{kWh}) \times \text{PUE} \times I_{\text{grid}} \;(\text{kgCO}_2\text{e/kWh})$$

- **$E_{\text{consumed}}$:** Direct hardware electrical energy measured via CodeCarbon / NVML board sensors or Apple Silicon SoC power registers.
- **$\text{PUE}$:** Data Center Power Usage Effectiveness (1.10 - 1.20 for modern hyperscale facilities; 1.60 for legacy enterprise).
- **$I_{\text{grid}}$:** Regional grid emissions intensity. Defaulted to the **India Central Electricity Authority (CEA) baseline of 0.708 kg CO2e / kWh**.

Dynamic time-varying carbon accounting:

$$\text{CO}_2\text{e} = \sum_{t=1}^{T} E_t \times I_t$$

Compounded multiplicative mitigation formulation:

$$\text{Final Carbon} = \text{Baseline Carbon} \times \prod_{i=1}^{k} \left(1 - S_i\right)$$

---

## Live Laptop Hardware Telemetry & Metal Acceleration

The project includes an integrated on-device hardware telemetry module (`utils/hardware.py`):

- **Host Platform Profile:** Apple MacBook Air (Apple M4, Model Mac16,13)
- **CPU Subsystem:** 10 Cores (4 Performance cores up to 4.4 GHz, 6 Efficiency cores up to 2.8 GHz)
- **Unified Memory:** 16 GB LPDDR5X (120 GB/s bandwidth) with real-time used/available memory monitoring
- **Accelerator:** Apple M4 10-Core GPU with Metal 4 API support
- **Neural Engine:** 16-Core Apple Neural Engine (38 TOPS)
- **Real-Time Telemetry:** Live CPU load %, per-core P/E utilization, battery state of charge, AC power mode, and estimated package wattage
- **On-Device Metal Benchmark:** Interactive PyTorch execution on `torch.device("mps")` measuring real forward-pass execution latency, energy in micro-Watt-hours ($\mu$Wh), and emissions avoidance in milligrams CO2e

---

## Installation & Local Execution

### Prerequisites
- Python 3.9+ (Python 3.11–3.13 recommended)
- macOS (Apple Silicon recommended for live Metal benchmarks) or Linux/Windows with CUDA

### 1. Clone the Repository
```bash
git clone https://github.com/Samcxx11/ai-emissions-dashboard.git
cd ai-emissions-dashboard
```

### 2. Set Up a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*Note: For Apple Silicon Metal GPU acceleration, ensure PyTorch with MPS support is installed:*
```bash
pip install torch psutil
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Automated Screenshot Generation

To capture full-page unconstrained screenshots (from header to footer with zero scroll cutoffs) across all 10 chapters:

```bash
# Ensure Brave Browser or Google Chrome is installed
python take_all_screenshots.py
```
Screenshots are automatically saved at high resolution to `~/Desktop/AI_Emissions_Dashboard_Screenshots/`.

---

## Research Literature & Foundations

1. **Strubell, E., Ganesh, A., & McCallum, A. (ACL 2019).** Energy and Policy Considerations for Deep Learning in NLP.
2. **Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (CACM 2020).** Green AI.
3. **Henderson, P., Hu, J., Romoff, J., et al. (JMLR 2020).** Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning.
4. **Patterson, D., Gonzalez, J., Le, Q., et al. (2021).** Carbon Emissions and Large Neural Network Training.
5. **Luccioni, A. S., Viguier, S., & Ligozat, A. L. (2022).** Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model.
6. **LeCun, Y. (2022).** A Path Towards Autonomous Machine Intelligence.
7. **Assran, M., Duval, Q., Misra, I., et al. (CVPR 2023).** Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).
8. **Bardes, A., Garrido, Q., Ponce, J., et al. (2024).** Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA).
9. **IEEE CAL (2024).** SLO-Aware GPU DVFS for Energy-Efficient LLM Inference Serving.
10. **Jiang, A. Q., et al. (2023).** Mistral 7B (Sliding Window Attention).
11. **vLLM Team (2024).** Automatic Prefix Caching Architecture & Design.

---

## License & Attribution

Developed by **Samarth Ralph** (`Samcxx11`).  
Released under the MIT License. Academic evaluations and benchmarking follow Open Science principles.
