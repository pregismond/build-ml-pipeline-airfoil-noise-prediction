# Build a Machine Learning Pipeline for Airfoil Noise Prediction

[![License](https://img.shields.io/badge/License-Apache_2.0-0D76A8?style=flat)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7.12](https://img.shields.io/badge/Python-3.7.12-green.svg)](https://shields.io/)

This repository contains my final project submission for **[IBM Skills Network - Coursera: Machine Learning with Apache Spark](https://www.coursera.org/learn/machine-learning-with-apache-spark)**

## Project Scenario

As a data engineer at an aeronautics consulting company, we take pride in our ability to efficiently design airfoils for use in both planes and sports cars. While our data scientists excel at Machine Learning, they rely on me to handle ETL (Extract, Transform, Load) tasks and construct ML pipelines.

## Objectives

* Clean the dataset
* Create a Machine Learning pipeline
* Evaluate the model's performance
* Persist it for future use

## Datasets

For this project, we will use a modified version of the NASA Airfoil Self-Noise dataset `NASA_airfoil_noise_raw.csv`, which is available in this repository.

The original dataset can be found here: NASA Airfoil Self-Noise dataset. https://archive.ics.uci.edu/dataset/291/airfoil+self+noise

The dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Notes

The metric values presented in the `Final_Project.ipynb` notebook can vary across different Python versions. These variations may occur due to changes in underlying libraries, algorithms, or default behavior. To ensure successful completion of the **Quiz: Final Project - Evaluation Submitted**, it is essential to complete this project using the Python version available in the Skill Network Labs (SN Labs) environment. 

## Usage

Install the required libraries using the provided `requirements.txt` file. The command syntax is:
```bash
python3 -m pip install -r requirements.txt
```

Download the required exchange rate file using the terminal command:
```bash
wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/datasets/NASA_airfoil_noise_raw.csv
```

Create folder to save model:
```bash
mkdir -p Final_Project
```

Execute the code using the command:
```bash
python3 Final_Project.py
```

## Learner

[Pravin Regismond](https://www.linkedin.com/in/pregismond)

## Instructor

[Ramesh Sannareddy](https://www.coursera.org/instructor/~75088416), Data Engineering Subject Matter Expert, @ IBM

## <h3 align="center"> © IBM Corporation 2023. All rights reserved. <h3/>
