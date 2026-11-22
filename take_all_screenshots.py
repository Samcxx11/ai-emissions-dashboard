"""
Complete Full-Page Screenshot Capture for AI Emissions Dashboard

Captures the entire page from top to bottom (unconstrained scroll height)
for all 10 dashboard chapters using Selenium + Brave Browser, ensuring
no cutoffs or truncated content.

Saves directly to ~/Desktop/AI_Emissions_Dashboard_Screenshots/
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

DESKTOP_DIR = os.path.expanduser("~/Desktop/AI_Emissions_Dashboard_Screenshots")
os.makedirs(DESKTOP_DIR, exist_ok=True)

PAGES = [
    ("00_Executive_Overview", "http://localhost:8501"),
    ("01_Energy_CO2_Comparison", "http://localhost:8501/Energy_CO2_Comparison"),
    ("02_Quantization_Impact", "http://localhost:8501/Quantization_Impact"),
    ("03_Per_Query_Analysis", "http://localhost:8501/Per_Query_Analysis"),
    ("04_Real_World_Impact", "http://localhost:8501/Real_World_Impact"),
    ("05_Batch_Size_Analysis", "http://localhost:8501/Batch_Size_Analysis"),
    ("06_Input_Length_Impact", "http://localhost:8501/Input_Length_Impact"),
    ("07_Carbon_Reduction_Strategies", "http://localhost:8501/Carbon_Reduction_Strategies"),
    ("08_JEPA_Carbon_Reduction", "http://localhost:8501/JEPA_Carbon_Reduction"),
    ("09_Inference_Optimization_Methods", "http://localhost:8501/Inference_Optimization_Methods"),
]

UNCONSTRAIN_CSS = """
    const style = document.createElement("style");
    style.innerHTML = `
        html, body {
            height: auto !important;
            overflow: visible !important;
        }
        [data-testid="stAppViewContainer"] {
            display: flex !important;
            flex-direction: row !important;
            height: auto !important;
            min-height: 100% !important;
            overflow: visible !important;
        }
        [data-testid="stSidebar"] {
            position: relative !important;
            height: auto !important;
            min-height: 100% !important;
            flex-shrink: 0 !important;
            width: 320px !important;
            min-width: 320px !important;
        }
        section.main, [data-testid="stMain"] {
            flex: 1 1 auto !important;
            height: auto !important;
            overflow: visible !important;
        }
        footer {
            display: none !important;
        }
    `;
    document.head.appendChild(style);
"""


def capture_all_fullpage():
    options = Options()
    options.binary_location = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--force-device-scale-factor=1.5")
    options.add_argument("--window-size=1680,1200")

    driver = webdriver.Chrome(options=options)
    print("Browser started. Capturing full-length screenshots for all chapters...")

    for filename, url in PAGES:
        print(f"\nProcessing: {filename} ({url})")
        driver.get(url)

        # Wait for Streamlit app to mount
        try:
            WebDriverWait(driver, 12).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stAppViewContainer']"))
            )
        except Exception:
            pass

        time.sleep(3.5)

        # For Chapter 09, trigger live Apple M4 benchmark so its results are included
        if "09_Inference_Optimization" in filename:
            try:
                buttons = driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    if "Execute Live Benchmark" in btn.text:
                        print("  Executing live Apple M4 Metal benchmark...")
                        btn.click()
                        time.sleep(4.0)
                        break
            except Exception as e:
                print(f"  Note: {e}")

        # Inject CSS to unconstrain heights for full-page capture
        driver.execute_script(UNCONSTRAIN_CSS)
        time.sleep(1.0)

        # Measure exact full content height
        total_height = driver.execute_script("""
            const main = document.querySelector('section.main') || document.querySelector('.block-container');
            return Math.max(
                main ? main.scrollHeight : 0,
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                1600
            );
        """)

        print(f"  Full content height: {total_height}px")
        driver.set_window_size(1750, total_height + 150)
        time.sleep(1.5)

        out_path = os.path.join(DESKTOP_DIR, f"{filename}.png")
        driver.save_screenshot(out_path)
        print(f"  Saved -> {out_path} ({os.path.getsize(out_path):,} bytes)")

    driver.quit()
    print("\nAll full-page screenshots captured successfully!")


if __name__ == "__main__":
    capture_all_fullpage()
