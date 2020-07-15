[![author](https://img.shields.io/badge/author-JohnnyLu-red.svg)](https://www.linkedin.com/in/jl1829/)
![GitHub](https://img.shields.io/github/license/jl1829/EmployeeSalaryPrediction)
[![Python 3.7](https://img.shields.io/badge/python-3.7.7-blue.svg)](https://www.python.org/downloads/release/python-377/)

EmployeeSalaryPrediction
==============================

# About
A short description of the project.


## Contents

1. [PlaceHolder](#1-ContentsHolderOne)
2. [PlaceHolder2](#2-How-to-Start)

## ContentsHolderOne

## Data Supplied

There's 3 CSV data files given:

* `train_features.csv` : Each row represents metadata for an individual job posting. The “jobId” column represents a unique identifier for the job posting. The remaining columns describe features of the job posting.
* `train_salaries.csv`: Each row associates a “jobId” with a “salary”.
* `test_features.csv`: Similar to `train_features.csv`, each row represents metadata for an individual job posting.

The first row of each file contains headers for the columns. Keep in mind that the metadata and salary data may contain errors.


## How to Start
placeholder


## Project Organization
------------
```bash
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
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   └── build_features.cpython-37.pyc
│   │   └── build_features.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── predict_model.cpython-37.pyc
│   │   │   └── predict_model_test.cpython-37.pyc
│   │   ├── predict_model.py
│   │   ├── predict_model_test.py
│   │   └── train_model.py
│   ├── preprocessing.py
│   └── visualization
│       ├── __init__.py
│       └── visualize.py
├── test_environment.py
└── tox.ini
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a></small></p>
