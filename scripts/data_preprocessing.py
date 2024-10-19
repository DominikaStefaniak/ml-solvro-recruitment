import pandas as pd
from utils import count_matching_dicts

dataframe = pd.read_json("C:/Users/domci/PycharmProjects/ml-solvro-recruitment/data/cocktail_dataset.json")

dataframe["alcoholic_ingredients"] = dataframe["ingredients"].apply(count_matching_dicts, args=("alcohol", 0))

dataframe['num_ingredients'] = dataframe['ingredients'].apply(len)

print(dataframe)
