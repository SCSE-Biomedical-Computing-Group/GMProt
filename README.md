# GMProt: Gated Modulation Protein Language Model for Prediction of Antimicrobial Peptide Activity

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

GMProt is a flexible pipeline for training, prediction, and visualization of feature contributions in antimicrobial peptide activity modeling, including the estimation of Minimum Inhibitory Concentration (MIC). This framework accompanies and supports our work, “Gated Protein Language Modeling for Accurate Prediction of Antimicrobial Peptide Activity.”

For a comprehensive description of the approach, please refer to the paper: [link here]

---

## 📑 Data Availability

The complete dataset is publicly available on **Zenodo**:
- **DOI:** https://doi.org/10.5281/zenodo.20740473  

### 📁 Folder Structure
```bash
├── alphafold_pdb_ecoli/
├── alphafold_pdb_stap/
├── fasta_ecoli/
├── fasta_stap/
├── five_fold_ecoli/
├── five_fold_s_aureus/
├── protT5/
└── grampa.csv
```
####  `alphafold_pdb_ecoli`
Predicted three-dimensional peptide structures (PDB format) targeting *Escherichia coli*, generated using **AlphaFold2**.
#### `alphafold_pdb_stap`
Predicted peptide structures (PDB format) targeting *Staphylococcus aureus*, generated using **AlphaFold2**.
####  `fasta_ecoli` and `fasta_stap`
FASTA-formatted peptide sequence datasets for:
- *E. coli*
- *S. aureus*
####  `five_fold_ecoli`
Preprocessed datasets stored in `.pkl` format for model training.
- Each file contains **five cross-validation splits**
- Multiple `.pkl` files represent alternative feature configurations
- See `readme.txt` for feature details
- `ecoli_dataset.pkl` is the final dataset used for **ModProt** training
#### `five_fold_s_aureus`
Five-fold cross-validation datasets for *S. aureus*.
####  `protT5`
Residue-level peptide embeddings computed using **ProtT5** for both:
- *E. coli*
- *S. aureus*
####  `grampa.csv`
The original antimicrobial peptide dataset containing sequences across multiple bacterial species, including:
- *E. coli*
- *S. aureus*
All derived datasets were constructed from this base file.

#### Dataset Attribution
The original `grampa.csv` dataset is attributed to:
> **Deep learning regression model for antimicrobial peptide design**  
> https://www.biorxiv.org/content/10.1101/692681v1.full
---

## Features

- **Train** the ModProt model on one or more datasets.  
- **Predict** activity values and evaluate model performance on test datasets.  
- **Plot** feature contributions to interpret model predictions.  
- Flexible handling of multiple datasets via indices.  
- Modular design for easy extension to other models and datasets.  

---

## Installation

1. **Clone the repository:**

```bash
git clone [URL]
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt 
```


3. **Usage**


```bash
python train.py <mode> [--model MODEL_NAME] [--datasets DATASET_INDICES]
```
| Argument     | Description                                                            |
| ------------ | ---------------------------------------------------------------------- |
| `mode`       | Required. Mode to run: `train`, `predict`, or `plot`.                  |
| `--model`    | Optional. Model name. Default: `GMProt`.                                |
| `--datasets` | Optional. List of dataset indices (supports multiple). Default: `[0]`. |

4. **Examples:** 
```bash
Train mode (default dataset [0]):
python train.py train
python train.py train --model Mymodel --datasets 0 1 2 3 4 

Predict mode 
python train.py predict --datasets 0 1 2 3 4

Plot feature contributions for specific datasets:
python train.py plot --datasets 0
```
Plots are saved to ./visualization/IG/fold_<dataset_index>.


5. **Directory Structure:**
```bash
├── train.py                  # Main script with argparse
├── model/                    # Pretrained or saved models
├── data/                     # Complte datasets
├── visualization/
│   └── IG/                   # Feature contribution plots
├── requirements.txt          # Python dependencies
└── README.md
```
Notes:

The mode argument is mandatory. Use one of train, predict, or plot.
For visualization, the model must have been trained and saved beforehand.
Ensure all dependencies and datasets are properly prepared before running.


## Citation
If you use this code, or dataset in your research, please cite our paper:

Prem Singh Bist, et al. (2026). **. Journal/Conference Name. DOI: 10.xxxx/xxxx
