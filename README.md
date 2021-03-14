# Current Research in Data Science
Working Repository for our project in Current Research in Data Science.

Team: Siddharth Bhargava & Tanveer Hannan

Final Documentation: https://www.overleaf.com/read/yspmvgqgxbsx


Data: https://www.kaggle.com/volodymyrgavrysh/bank-marketing-campaigns-dataset


## Setting Up the Environment

**Create Python Environment**

Run the following command in the terminal.
```
  conda create -n iml_env python=3.8.5
  conda activate iml_env
  pip install -r requirements.txt
  pip install 'ray[tune]'
```

## Running our Code

### Step 1: Preprocessing

Run the following command to created the imputed data. The path to the original data file and the type of imputer has been defined inside the script.
```
run src/imputation.py
```
Our pre-processing steps have been encapsulated in our DataGenerator object defined in our data_generator.py script. For visualizations related to our exploratory data analysis, please refer to our Notebook "pre_processing.ipynb" (Notebooks/pre_processing.ipynb)

### Step 2: Modeling

Run the following command to train and tune our models. The different hyperparamaters for each chosen model have been defined within the script.
```
run src/hpo_all_models.py
```
For data, we load our imputed data file (data/imputed_bank_data_mice.csv) into DataGenerator module, called in the script. Our evaluations can be seen in the Notebook "evaluate_models.ipynb" (Notebooks/evaluate_models.ipynb)

### Step 3: Interpretable Machine Learning

We defined our counterfactual module in the counterfactuals.py script. 

For our analysis based on shapley values, please refer to our Notebook "SHAP.ipynb" and for our analysis using Counterfactuals, please refer to our notebook "counterfactuals.ipynb".
