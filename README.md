[![author](https://img.shields.io/badge/author-JohnnyLu-red.svg)](https://www.linkedin.com/in/jl1829/)
![GitHub](https://img.shields.io/github/license/jl1829/EmployeeSalaryPrediction)
[![Python 3.7](https://img.shields.io/badge/python-3.7.7-blue.svg)](https://www.python.org/downloads/release/python-377/)

EmployeeSalaryPrediction
==============================
![image](https://raw.githubusercontent.com/JL1829/EmployeeSalaryPrediction/master/imgs/job-search.png)


# About
Can you provide a predictive indication of a candidate's salary level, based on their previous job description mata data? The Employee Salary Prediction is a sub project for a complex HR system to evaluate the candidate's expected salary level, based on the actual job meta data and salary mapping dataset. 


## Contents

1. [Origin Business Problem](#1-Origin-Business-Problem)
2. [Dataset description](#2-Dataset-description)
3. [How to Start](#3-How-to-start)
4. [Exploration Notebook](#4-Exploration-Notebook)
5. [Project Organization](#5-Project-Organization)

### Origin Business Problem
Customer is a large recruiting firm have more than millions record of the candidates and tens of thousand of reviews from the candidates to each particular companys and the companys to candidates. Cusomter was frustrated about the miss match of the salary expectation from both company and candidate, but they managed to dig out the pass record of success recruiting case, based on this situation, customer want to develop a predictive salary indication measurement to improve the expectation of both candidate and companys. 

### Dataset description

There's 3 CSV data files given:

* `train_features.csv` : Each row represents metadata for an individual job posting. The “jobId” column represents a unique identifier for the job posting. The remaining columns describe features of the job posting.
* `train_salaries.csv`: Each row associates a “jobId” with a “salary”.
* `test_features.csv`: Similar to `train_features.csv`, each row represents metadata for an individual job posting.

The first row of each file contains headers for the columns. Keep in mind that the metadata and salary data may contain errors.


## How to Start

```bash
> git clone https://github.com/JL1829/EmployeeSalaryPrediction.git
> cd EmployeeSalaryPrediction
> pip3 install -r requirement.txt
> python3 main.py
```

## Exploration Notebook


## Project Organization
------------
```bash
.
├── LICENSE
├── Makefile
├── README.md
├── Testing.ipynb
├── data
│   ├── external
│   ├── internal
│   ├── processed
│   │   └── processed.csv
│   └── raw
│       ├── test_features.csv
│       ├── testing.csv
│       ├── train_features.csv
│       └── train_salaries.csv
├── docs
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── imgs
│   ├── job-search.png
│   └── sample.png
├── main.py
├── models
├── notebooks
│   ├── EDA.ipynb
│   └── Modeling.ipynb
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── predict_model.py
│   │   ├── predict_model_test.py
│   │   └── train_model.py
│   ├── preprocessing.py
│   └── visualization
│       ├── __init__.py
│       ├── visualize.py
│       └── visuzlize_Testing.py
├── test_environment.py
└── tox.ini
```

## To Do:
- Streamlit Web App
- Put it online
