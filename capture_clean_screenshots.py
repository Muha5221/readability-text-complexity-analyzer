"""
Generate clean, well-focused presentation screenshots of all sections of the application.
"""
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

SCREENSHOT_DIR = Path(__file__).resolve().parent / "Readability_Text_Complexity_Analyzer" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def run():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1440,1050")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get("http://localhost:8501")
        time.sleep(5)

        # 1. Landing state
        driver.save_screenshot(str(SCREENSHOT_DIR / "01_landing_page.png"))
        print("Captured 01_landing_page.png")

        # 2. Click Medium sample
        for b in driver.find_elements("tag name", "button"):
            if "Medium" in b.text:
                driver.execute_script("arguments[0].click();", b)
                break
        time.sleep(4)

        # Overview & Readability
        driver.save_screenshot(str(SCREENSHOT_DIR / "02_overview_metrics_readability.png"))
        print("Captured 02_overview_metrics_readability.png")

        # Scroll to Lexical Analysis & POS Chart
        driver.execute_script("""
            var el = document.querySelector('[data-testid="stAppViewContainer"]') || document.documentElement;
            el.scrollTop = 700;
        """)
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOT_DIR / "03_lexical_and_pos_analysis.png"))
        print("Captured 03_lexical_and_pos_analysis.png")

        # Scroll to WordNet Explorer & Similarity Tab
        driver.execute_script("""
            var el = document.querySelector('[data-testid="stAppViewContainer"]') || document.documentElement;
            el.scrollTop = 1450;
        """)
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOT_DIR / "04_wordnet_explorer.png"))
        print("Captured 04_wordnet_explorer.png")

        # Click Word Similarity Tab
        for tab in driver.find_elements("xpath", "//button[contains(., 'Word Similarity')]"):
            driver.execute_script("arguments[0].click();", tab)
            time.sleep(2)
            break

        # Click Calculate WordNet Similarity
        for btn in driver.find_elements("xpath", "//button[contains(., 'Calculate WordNet Similarity')]"):
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(3)
            break

        driver.save_screenshot(str(SCREENSHOT_DIR / "05_wordnet_similarity_result.png"))
        print("Captured 05_wordnet_similarity_result.png")

        # Scroll to Final Analysis
        driver.execute_script("""
            var el = document.querySelector('[data-testid="stAppViewContainer"]') || document.documentElement;
            el.scrollTop = 2200;
        """)
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOT_DIR / "06_final_linguistic_analysis.png"))
        print("Captured 06_final_linguistic_analysis.png")

        print("Finished capturing all presentation screenshots!")
    finally:
        driver.quit()

if __name__ == "__main__":
    run()
