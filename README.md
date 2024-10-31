# Cocktail Clustering Project

## Overview
It's a recruitment project for the Solvro Scientific Club at Wrocław University of Science and Technology. It focuses on exploratory data analysis (EDA) and clustering of a cocktail dataset. The task involves conducting EDA, cleaning and preprocessing the data and performing clustering using **KMeans** and **DBSCAN** to identify unique clusters of cocktails based on their ingredients and other characteristics.

## Data 📋
The dataset contains cocktails with associated ingredient lists and characteristics. It's stored in a `.json` file located in the `data` folder. This is loaded and processed into a DataFrame format for easy manipulation and analysis using Pandas.  

## Repository Structure
```data/:``` Raw and processed datasets.  
```scripts/:``` Scripts for data preprocessing and clustering.  
```notebooks/:``` Jupyter notebooks for EDA, cleaned data visualization and clustering visualization.  
```requirements.txt:``` Dependency management files. 

## Quick start ⚡
To replicate the experiments, follow steps listed below (in cmd).  
As an alternative you can do it using conda.

### 1. Clone the Repository
```bash
git clone https://github.com/DominikaStefaniak/ml-solvro-recuitemnt.git
cd ml-solvro-recruitment
```

### 2. Set up the environment and install libraries
If you want to do it in cmd:  
First create venv:
```
python -m venv venv
```
Activate venv:  
**For windows:**
```
venv\Scripts\activate
```
**For macOS and Linux:**
```
source venv/bin/activate
```
Install the dependecies from requirements.txt  
**note:** the library versions are compatible with python 3.12
```
pip install -r requirements.txt
```

#### Possible issues:
- in venv "bin" file created instead of "Scripts" for windows: use command prompt or power shell, if you are then python may have been installed incorectly
- error while libraries instalation: make sure to use python 3.12 or check the libraries versions compatible for your python version and change them in requirements file

### 3. Steps of the experiments  
Remember to always delete corresponding data file before running any of the .py files  
- data analysis in **eda.ipynb** file
- data cleaning and preprocessing in **data_preprocessing.py** 
- data visualization in **data_visualization.ipynb**
- additional preprocesing and clustering models with quantitative evaluation in **clustering.py**
- qualtitative evaluations with summary in **clustering_results.ipynb**
  
