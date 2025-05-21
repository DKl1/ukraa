![Project Logo](ukraa.png)
# Ukrainian Auto Alignment Library | UKRAA
Ukrainian Auto Alignment Library (UKRAA) is a Python library for automatic text alignment. Designed primarily for Ukrainian as the core language, it supports alignment between many language pairs using flexible data loading, multiple embedding models, and advanced evaluation metrics.

----------
## Installation

### Prerequisites
    - Python 3.10 (recommended)
    - Conda or a virtual environment to isolate dependencies

#### Using Conda
1. Create and Activate Environment:
    ``` bash
        conda create -n ukraa_conda python=3.10
        conda activate ukraa_conda
    ```

2. Clone the Repository:
    ```bash
        git clone https://github.com/yourusername/ukraa.git
        cd ukraa
    ```

3. Install the Package in Editable Mode:
    ```bash
      pip install -e .
    ```



#### Using Virtualenv
1.	Create Environment:
```bash
    python3.10 -m venv ukraa_env
```
2.  Activate Environment:
On MacOs / Linux:
```bash
  source ukraa_env/bin/activate  
```
On Windows
```bash
    ukraa_env\Scripts\activate
```

3. Clone the Repository and Install:
```bash
    git clone https://github.com/yourusername/ukraa.git
    cd ukraa
    pip install -e .
```

### Models Installation
To download lasers models:
```bash
python -m laserembeddings download-models
```


---
## Usage

The library provides a command-line interface (CLI) through the auto-align entry point.

##### Example Command

To align sentences between a source file and a target file (with an optional gold standard file for evaluation), run:

```bash
auto-align --src-file data/uk.txt --tgt-file data/en.txt --gold data/gold.txt --src-lang uk --tgt-lang en --topk 5
```

##### CLI Parameters

- **Required Parameters:**
  - `--src-file` / `-s`: Path to the source text file
  - `--tgt-file` / `-t`: Path to the target text file

- **Language Settings:**
  - `--src-lang` / `-sl`: Source language code (e.g., "uk" for Ukrainian)
  - `--tgt-lang` / `-tl`: Target language code (e.g., "en" for English)

- **Alignment Controls:**
  - `--threshold` / `-th`: Cosine similarity threshold (0 to 1). Default: 0.7
  - `--topk` / `-k`: Number of nearest neighbors to consider for each source sentence. Default: 5
  - `--batch-size` / `-b`: Batch size for processing embeddings. Default: 512

- **Model Selection:**
  - `--encoder` / `-e`: Which encoder to use. Options: "labse", "laser", "laser2", "sbert"

- **Output and Evaluation:**
  - `--output` / `-o`: Output file path for aligned pairs. Default: 'aligned_output.txt'
  - `--gold` / `-g`: Path to gold alignment file (for evaluation)
  - `--verbose` / `-v`: Enable verbose logging (debug mode)

##### Example Commands

**High precision alignment:**
```bash
auto-align --src-file data/uk.txt --tgt-file data/en.txt --threshold 0.85 --topk 3
```

**High recall alignment:**
```bash
auto-align --src-file data/uk.txt --tgt-file data/en.txt --threshold 0.6 --topk 10
```

**Using a specific encoder:**
```bash
auto-align --src-file data/uk.txt --tgt-file data/en.txt --encoder labse
```

**With evaluation against gold standard:**
```bash
auto-align --src-file data/uk.txt --tgt-file data/en.txt --gold data/gold.txt
```

##### Output
- **Aligned Output:**
  The aligned pairs, along with cosine similarity scores, are saved to aligned_output.txt (or the path specified by --output).
- **Evaluation Metrics:**
If a gold standard file is provided, evaluation metrics (precision, recall, F1, TER, BLEU, CHRF, BERT-Score) are computed and appended to the output file.

---
