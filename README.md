# Readability & Text Complexity Analyzer

Analyze text readability, vocabulary and sentence structure.

---

## Overview

Readability & Text Complexity Analyzer is a lightweight, responsive web application for analyzing English text readability, vocabulary difficulty, sentence structures, and grammatical distributions using Natural Language Processing (NLTK).

The application evaluates:
- **Text Statistics:** Word count, sentence count, unique words, average sentence length, and average word length.
- **Readability Scores:** Flesch Reading Ease and Flesch-Kincaid Grade Level with practical reading interpretations.
- **Vocabulary Analysis:** Lexical diversity (Type-Token Ratio), average word length, long word detection, and word frequency distribution.
- **Part-of-Speech Analysis:** Distribution of nouns, verbs, adjectives, and adverbs with visual charts.
- **Final Analysis:** Practical, rule-based observations on overall text complexity (Easy, Medium, or Difficult).

---

## Folder Structure

```text
Readability_Text_Complexity_Analyzer/
│
├── app.py                      # Main Streamlit web dashboard
├── requirements.txt            # Python dependencies
├── nltk.txt                    # NLTK data resources for cloud deployment
├── README.md                   # Documentation and user guide
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py       # Sentence and word tokenization and cleaning
│   ├── lexical_analysis.py    # Type-Token Ratio, word frequencies, and long words
│   ├── pos_analysis.py        # Part-of-Speech tagging and category distribution
│   ├── wordnet_analysis.py    # WordNet synsets, definitions, and synonyms
│   ├── readability.py         # Syllable heuristic, FRE, and FKGL formulas
│   └── complexity.py          # Complexity classification and final observations
│
└── sample_data/
    ├── easy_text.txt          # Sample conversational narrative
    ├── medium_text.txt        # Sample balanced expository text
    └── difficult_text.txt     # Sample advanced technical text
```

---

## Installation & Local Execution

### Requirements
- Python 3.9+
- pip

### Setup

1. **Navigate to the Project Directory:**
   ```bash
   cd Readability_Text_Complexity_Analyzer
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   *(Or run `python -m streamlit run app.py`)*

   The application will open in your browser at `http://localhost:8501`.

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository, set branch to `main`, and main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and automatically caches required NLTK resources listed in `nltk.txt`.

---

## Readability Formulas

- **Flesch Reading Ease (FRE):**
  $$\text{FRE} = 206.835 - (1.015 \times \text{ASL}) - (84.6 \times \text{ASW})$$
  *Where ASL is Average Sentence Length and ASW is Average Syllables per Word.*
- **Flesch-Kincaid Grade Level (FKGL):**
  $$\text{FKGL} = (0.39 \times \text{ASL}) + (11.8 \times \text{ASW}) - 15.59$$
