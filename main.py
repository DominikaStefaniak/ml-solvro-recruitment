import pandas as pd

from preprocessing_functions import count_matching_dicts

dataframe = pd.read_json("C:/Users/domci/PycharmProjects/ml-solvro-recruitment/cocktail_dataset.json")

dataframe["alcoholic_ingredients"] = dataframe["ingredients"].apply(count_matching_dicts, args=("alcohol", 1))

print(dataframe)
