# 🌍 AI Inference Emissions Dashboard

An interactive visualization project that measures the real-world environmental cost of running AI language models. It takes technical benchmarking data (like Joules and kWh) and translates it into easy-to-understand metrics (like car miles driven and smartphone charges) to promote the use of right-sized, energy-efficient AI models.

## 🚀 Features

- **Model Comparison:** Compare energy and CO₂ costs across different sizes of models (1.1B up to 13B).
- **Quantization Impact:** See how compressing a model (FP16, INT8, INT4) affects energy consumption and response quality.
- **Batch Size & Input Length Analysis:** Understand how processing multiple prompts at once or changing prompt lengths changes the energy cost per token.
- **Real-World Calculator:** Estimate the annual environmental impact based on daily query volume.

## 💻 How to Run on Your Laptop

Follow these steps to run the interactive dashboard locally on your machine.

### Prerequisites
Make sure you have [Python 3.9+](https://www.python.org/downloads/) installed on your system.

### 1. Clone the repository
Open your terminal and clone the project:
```bash
git clone https://github.com/Samcxx11/ai-emissions-dashboard.git
cd ai-emissions-dashboard
```

### 2. Create a Virtual Environment (Recommended)
It is good practice to use a virtual environment to manage dependencies.
```bash
# On Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages (Streamlit, Pandas, Plotly, etc.):
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```

After running this command, your browser should automatically open the dashboard at `http://localhost:8501`.

---

## 🛠️ Data Pipeline & CodeCarbon
The repository also includes a `benchmark` folder with scripts to run actual energy measurements using **CodeCarbon**.
If you have a GPU and want to run your own experiments instead of using the pre-measured dataset:
```bash
python -m benchmark.run_inference --experiment all
```

## 📚 Acknowledgments
Methodology inspired by research from Strubell et al. (ACL 2019) and Henderson et al. (JMLR 2020).
