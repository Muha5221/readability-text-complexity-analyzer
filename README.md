# Readability and Text Complexity Analyzer Using Natural Language Processing (NLTK)

**M.Tech Computational Linguistics Course Project**  
*Modules Covered:* Module 1 (Introduction to Language Processing) & Module 2 (Processing and Understanding Natural Language)

---

## 📌 Project Overview

The **Readability and Text Complexity Analyzer** is a lightweight, fully functional web application designed to evaluate English text across multiple linguistic dimensions:
1. **Structural & Surface Readability:** Sentence length, syllable counts, Flesch Reading Ease (FRE), and Flesch-Kincaid Grade Level (FKGL).
2. **Lexical Characteristics:** Lexical diversity (Type-Token Ratio), average word length, long word distributions, and vocabulary rarity comparison using the standard NLTK Brown reference corpus.
3. **Grammatical & Morphosyntactic Distribution:** Part-of-Speech (POS) tagging and distribution of content words (Nouns, Verbs, Adjectives, Adverbs).
4. **Lexical Semantics & Ontological Knowledge:** WordNet synset definitions, synonym discovery, and WordNet taxonomic path similarity between word pairs.
5. **Rule-Based Complexity Classification:** Transparent, deterministic classification into **Easy**, **Medium**, or **Difficult** without opaque machine learning or external black-box AI APIs.

---

## 🏗️ Project Architecture & Folder Structure

```text
Readability_Text_Complexity_Analyzer/
│
├── app.py                      # Main Streamlit web dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation and project guide
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py       # Sentence/word tokenization, cleaning, normalization
│   ├── lexical_analysis.py    # TTR, NLTK FreqDist, Brown corpus comparison, long words
│   ├── pos_analysis.py        # NLTK pos_tagging, grammatical category aggregation
│   ├── wordnet_analysis.py    # WordNet synsets, synonyms, and path similarity
│   ├── readability.py         # Syllable heuristic, FRE, and FKGL formulas
│   └── complexity.py          # Multi-criteria scoring & rule-based explanation
│
├── sample_data/
│   ├── easy_text.txt          # Elementary narrative sample (High readability, simple syntax)
│   ├── medium_text.txt        # Expository prose sample (Standard syntax and vocabulary)
│   └── difficult_text.txt     # Dense academic sample (Complex syntax, specialized terminology)
│
└── screenshots/               # Actual application execution screenshots
```

---

## 📖 Computational Linguistics Syllabus Alignment

### Module 1: Introduction to Language Processing
- **Sentence Tokenization:** Divides prose into sentence units using NLTK `sent_tokenize`.
- **Word Tokenization:** Segments text into individual lexical tokens using NLTK `word_tokenize`.
- **Token Filtering & Normalization:** Strips punctuation marks and converts tokens to lower case to eliminate statistical distortion.
- **Part-of-Speech (POS) Tagging:** Classifies words into grammatical categories using the Penn Treebank tagset via NLTK `pos_tag`.
- **Frequency Distribution:** Computes empirical word occurrences and ranking using NLTK `FreqDist`.
- **Tagged Text Analysis:** Aggregates and quantifies the proportion of nominal, verbal, adjectival, and adverbial heads.

### Module 2: Processing and Understanding Natural Language
- **WordNet Lexical Database:** Queries WordNet synsets (`wn.synsets`) to retrieve definitions, grammatical categories, and contextual usage examples.
- **Synonym Extraction:** Identifies true lexical synonyms using WordNet lemmas without automated generative hallucinations.
- **Semantic Similarity:** Computes shortest path distance between concepts in WordNet's hypernym/hyponym hierarchy (`synset.path_similarity`).
- **Corpus Linguistics & Reference Comparison:** Compares text vocabulary against the balanced **NLTK Brown Corpus** (~1 million words) to identify infrequent or domain-specific terminology.

---

## 🧮 Mathematical Formulas and Documented Heuristics

### 1. Documented Syllable Counting Heuristic
Rather than relying on closed-source or black-box libraries, syllables are counted using a transparent rule-based algorithm:
1. Normalize word to lower-case alphabetic characters.
2. If word length $\le 3$, return $1$ syllable (e.g., *the*, *sun*, *dog*).
3. Identify continuous vowel groups `[aeiouy]+` as syllable nuclei.
4. Subtract $1$ for silent ending `-e` (e.g., *make*, *drive*), except when part of `-le` preceded by a consonant (e.g., *ta-ble*) or ending in `-ee` (e.g., *tree*).
5. Subtract $1$ for silent inflectional endings `-ed` (not preceded by *t* or *d*) and `-es`.
6. Enforce a minimum floor of $1$ syllable per word.

### 2. Readability Formulas
- **Average Sentence Length (ASL):**
  $$\text{ASL} = \frac{\text{Total Clean Words}}{\text{Total Sentences}}$$
- **Average Syllables per Word (ASW):**
  $$\text{ASW} = \frac{\text{Total Syllables}}{\text{Total Clean Words}}$$
- **Flesch Reading Ease (FRE):**
  $$\text{FRE} = 206.835 - (1.015 \times \text{ASL}) - (84.6 \times \text{ASW})$$
  *Interpretation:* $\ge 70$: Easy | $50 - 69.99$: Moderate | $< 50$: Difficult.
- **Flesch-Kincaid Grade Level (FKGL):**
  $$\text{FKGL} = (0.39 \times \text{ASL}) + (11.8 \times \text{ASW}) - 15.59$$
  *Interpretation:* Represents the estimated U.S. school grade level required for comprehension.

### 3. Multi-Criteria Complexity Classification
The text is evaluated across 5 transparent linguistic indicators. Each indicator is awarded points ($1 = \text{Easy}$, $2 = \text{Medium}$, $3 = \text{Difficult}$):

| Linguistic Indicator | Easy (1 pt) | Medium (2 pts) | Difficult (3 pts) |
| :--- | :--- | :--- | :--- |
| **Average Sentence Length (ASL)** | $< 15$ words | $15 - 24$ words | $\ge 25$ words |
| **Average Word Length (AWL)** | $< 4.60$ chars | $4.60 - 5.40$ chars | $> 5.40$ chars |
| **Lexical Diversity (TTR)** | $< 0.55$ | $0.55 - 0.74$ | $\ge 0.75$ |
| **Less Frequent Vocabulary Ratio** | $< 6.0\%$ | $6.0\% - 15.99\%$ | $\ge 16.0\%$ |
| **Flesch Reading Ease (FRE)** | $\ge 70.0$ | $50.0 - 69.99$ | $< 50.0$ |

$$\text{Composite Score} = \frac{\sum \text{Indicator Points}}{5}$$

- $\text{Composite Score} < 1.67 \implies \mathbf{Easy}$
- $1.67 \le \text{Composite Score} < 2.34 \implies \mathbf{Medium}$
- $\text{Composite Score} \ge 2.34 \implies \mathbf{Difficult}$

---

## 🚀 Installation & Local Execution

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Internet access during initial run (to download NLTK tokenizers and WordNet if not cached)

### Setup Instructions

1. **Clone or Navigate to the Project Directory:**
   ```bash
   cd Readability_Text_Complexity_Analyzer
   ```

2. **(Optional) Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```
   The interactive dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

This application is 100% cloud-ready and contains self-contained resource loaders.

1. Push this repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"**.
4. Select your GitHub repository, branch (`main`), and set the main file path to:
   ```text
   Readability_Text_Complexity_Analyzer/app.py
   ```
   (or `app.py` if deployed from the repository root).
5. Click **"Deploy"**. Streamlit Cloud will install `requirements.txt` and run the application.

---

## 🧪 Demonstration & Presentation Guide

For a 5–10 minute academic evaluation:
1. **Show Sample 1 (Easy):** Click **🟢 Easy** in the sidebar. Point out short sentence length ($< 10$ w/s), elementary vocabulary ($0\%$ rare words), high Flesch Reading Ease ($> 85$), and **Easy** classification.
2. **Show Sample 2 (Medium):** Click **🟡 Medium** in the sidebar. Highlight balanced syntactic structures (~20 w/s), intermediate vocabulary ($~8\%$ rare words), and **Medium** classification.
3. **Show Sample 3 (Difficult):** Click **🔴 Difficult** in the sidebar. Point out long sentences ($> 20$ w/s), high nominal density ($> 45\%$ nouns), polysyllabic academic terminology ($> 30\%$ long words), low Reading Ease ($< 25$), and **Difficult** classification.
4. **Demonstrate WordNet Explorer:**
   - Look up a word like *"syntactic"* or *"algorithm"* to show synsets, definitions, and synonyms.
   - Run a path similarity comparison between *"algorithm"* and *"computer"* (high score) vs *"algorithm"* and *"banana"* (no taxonomic path or low score).
5. **Explain Section 7 Final Analysis:** Highlight how the summary is systematically constructed from empirical indicators without generative black-box models.
