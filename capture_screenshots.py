"""
Automated Screenshot Capture Script using Headless Chrome & Selenium.
Captures all major sections of the Readability & Text Complexity Analyzer:
1. 01_initial_landing.png
2. 02_easy_text_analysis.png
3. 03_medium_text_analysis.png
4. 04_difficult_text_analysis.png
5. 05_wordnet_explorer.png
6. 06_wordnet_similarity.png
7. 07_final_analysis_summary.png
"""

import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

SCREENSHOTS_DIR = Path(__file__).resolve().parent / "Readability_Text_Complexity_Analyzer" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def capture_ui():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1440,1100")

    driver = webdriver.Chrome(options=options)

    try:
        url = "http://localhost:8501"
        print(f"Opening {url}...")
        driver.get(url)
        time.sleep(5)

        # 1. Capture Initial State
        driver.save_screenshot(str(SCREENSHOTS_DIR / "01_initial_landing.png"))
        print("Captured 01_initial_landing.png")

        # 2. Click Medium Sample
        for b in driver.find_elements(By.TAG_NAME, "button"):
            if "Medium" in b.text:
                b.click()
                print("Clicked Medium button...")
                break
        time.sleep(4)

        # Overview and Readability
        driver.save_screenshot(str(SCREENSHOTS_DIR / "02_metrics_and_readability.png"))
        print("Captured 02_metrics_and_readability.png")

        # Scroll to Lexical and POS
        driver.execute_script("window.scrollTo(0, 950);")
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "03_lexical_and_pos_analysis.png"))
        print("Captured 03_lexical_and_pos_analysis.png")

        # Scroll to Section 6 WordNet
        driver.execute_script("window.scrollTo(0, 1650);")
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "04_wordnet_explorer.png"))
        print("Captured 04_wordnet_explorer.png")

        # Click Similarity Tab and Calculate
        for tab in driver.find_elements(By.XPATH, "//button[contains(., 'Word Similarity')]"):
            driver.execute_script("arguments[0].click();", tab)
            time.sleep(2)
            break

        for btn in driver.find_elements(By.XPATH, "//button[contains(., 'Calculate WordNet Similarity')]"):
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(3)
            break

        driver.save_screenshot(str(SCREENSHOTS_DIR / "05_wordnet_similarity_result.png"))
        print("Captured 05_wordnet_similarity_result.png")

        # Scroll to Section 7 Final Analysis
        driver.execute_script("window.scrollTo(0, 2200);")
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "06_final_analysis_summary.png"))
        print("Captured 06_final_analysis_summary.png")

        print("All high-definition screenshots successfully captured!")

    finally:
        driver.quit()

if __name__ == "__main__":
    capture_ui()
