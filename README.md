# AI-Augmented Estimation

This repository contains code and experiments for results appeared in 
    [Wang, Mengxin, Dennis J. Zhang, and Heng Zhang. "Large language models for market research: A data-augmentation approach." arXiv preprint arXiv:2412.19363 (2024).](https://arxiv.org/pdf/2412.19363)


## Requirements
The project uses Python 3.11.7 and requires the packages in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mxw-selene/ai-augmented-estimation.git
cd ai-augmented-estimation
```

2. Create and activate a conda environment:
```bash
conda create -n aae python=3.11.7
conda activate aae
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── vaccine/                    # Empirical Study in Section 5
├── car/                        # Empirical Study in Section 6.1
├── aae-vs-ppi/                 # Empirical Study in Section 6.2
└── model-misspecification/     # Empirical Study in Section 6.3
```

```
vaccine/
├── data/                   # Dataset storage
├── res/                    # Saved experimental results
├── notebooks/              # Jupyter notebook for experiment results
├── exp.sh                  # Bash script for running experiments in batch
├── debias.py               # Script for running AAE experiments
├── naive.py                # Script for running naive estimation experiments
├── ppi.py                  # Script for running PPI experiments
├── estimation_weighted.py  # Utility functions for estimation
├── ppi_utils.py            # Utility functions for PPI
└── ground_truth.py         # Script for estimating the ground truth parameters
```

```
car/
├── data/                   # Dataset storage
├── res/                    # Saved experimental results
├── notebooks/              # Jupyter notebook for experiment results
├── exp.sh                  # Bash script for running experiments in batch
├── debias.py               # Script for running AAE experiments
├── naive.py                # Script for running naive estimation experiments
├── estimation_weighted.py  # Utility functions for estimation
└── ground_truth.py         # Script for estimating the ground truth parameters
```


```
aae-vs-ppi/
├── data/                   # Dataset storage
├── res/                    # Saved experimental results
├── aae.py                  # AAE functions
├── ppi_utils.py            # PPI utility functions
├── estimation_weighted.py  # Utility functions for estimation
└── section_6.2.ipynb       # Jupyter notebook for experiment results
```

```
model-misspecification/
├── data/                   # Dataset storage
├── res/                    # Saved experimental results
├── aae.py                  # AAE functions
├── estimation_weighted.py  # Utility functions for estimation
└── section_6.3.ipynb       # Jupyter notebook for experiment results
```



## Usage

You can run the experiments and reproduce the results for each section as follows:

### Section 5
- Use `vaccine/notebooks/section_5.ipynb` to produce the results presented in Section 5. These results are based on experimental data stored in `vaccine/res/`.

- To re-run all experiments from scratch, execute the following command in the terminal:

    ```
    bash exp.sh
    ```


    The program will prompt you to confirm whether you want to overwrite existing results in `vaccine/res/`. It may take several hours to complete all the experiments.
    It is recommended to back up the existing results before proceeding.

### Section 6.1
- Use `car/notebooks/section_6.1.ipynb` to produce the results presented in Section 5. These results are based on experimental data stored in `car/res/`.

- To re-run all experiments from scratch, execute the following command in the terminal:

    ```
    bash exp.sh
    ```


    The program will prompt you to confirm whether you want to overwrite existing results in `car/res/`. It may take several hours to complete all the experiments.
    It is recommended to back up the existing results before proceeding.

### Section 6.2

- Use `aae-vs-ppi/section_6.2.ipynb` to reproduce the experiments in Section 6.2.

### Section 6.3

- Use `model-misspecification/section_6.3.ipynb` to reproduce the experiments in Section 6.3.



## Citation

If you use this code or data in your research, please cite our paper:

'''
Wang, Mengxin, Dennis J. Zhang, and Heng Zhang. "Large language models for market research: A data-augmentation approach." arXiv preprint arXiv:2412.19363 (2024).
'''

## License
| Component        | License                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Codebase         | [MIT License](LICENSE)                                                                                                                      |
| Datasets         | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)                 |


## Contact

[Mengxin Wang](https://mxwang.site)

Email: mengxin dot wang at utdallas dot edu
