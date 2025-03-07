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
auto-align --src_file data/uk.txt --tgt_file data/en.txt --gold_file data/gold.txt --src_lang uk --tgt_lang en --k 5
```

- --src_file: Path to the source text file.
- --tgt_file: Path to the target text file.
- --gold_file: (Optional) Path to the gold standard indices file.
- --src_lang: Source language code (default is "uk" for Ukrainian).
- --tgt_lang: Target language code (default is "en" for English).
- --k: Number of nearest neighbors (default is 5).

##### Output
- **Aligned Output:**
  The aligned pairs, along with FAISS distances, are saved to aligned_output.txt.
- **Evaluation Metrics:**
If a gold standard file is provided, evaluation metrics (e.g., precision@1, precision@k, MRR) are computed and printed to the console.

---
## Contributing

Contributions are welcome! If you’d like to contribute:
1. Fork the repository. 
2. Create a feature branch.
3. Submit a pull request.

Please follow PEP8 guidelines and ensure your code includes comprehensive docstrings and type hints.
