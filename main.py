import pandas as pd

from preprocessing_functions import column_count_dicts_with_condition

dataframe = pd.read_json("C:/Users/domci/ml-solvro-recruitment1/cocktail_dataset.json")

df = column_count_dicts_with_condition(dataframe, "ingredients", "alcohol", 1, "alcoholic_ingredients")

print(df)
