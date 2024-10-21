import pandas as pd
from utils import count_matching_dicts
from utils import generalise_items_in_column
from sklearn.preprocessing import OneHotEncoder


#import data using pandas
dataframe = pd.read_json("C:/Users/domci/PycharmProjects/ml-solvro-recruitment/data/cocktail_dataset.json")

#change categorical columns to category datatype
dataframe = dataframe.astype({
    "category": "category",
    "glass": "category"
})

#encode category column to binary column "ordinaryDrink"
dataframe["ordinaryDrink"] = dataframe["category"].apply(lambda category: 1 if category == "Ordinary Drink" else 0)
dataframe = dataframe.drop("category", axis=1)

#generalise glass type's that occurred less than 3 times
dataframe = generalise_items_in_column(dataframe, "glass", 3, "Other")

#encode generalised glass column
enc = OneHotEncoder(sparse_output=False)
encoded_glass = enc.fit_transform(dataframe[["glass"]])
encoded_glass_df = pd.DataFrame(encoded_glass, columns=enc.get_feature_names_out(['glass']))
dataframe = pd.concat([dataframe, encoded_glass_df], axis=1)
dataframe = dataframe.drop('glass', axis=1)


#dataframe["alcoholic_ingredients"] = dataframe["ingredients"].apply(count_matching_dicts, args=("alcohol", 0))

#dataframe['num_ingredients'] = dataframe['ingredients'].apply(len)

print(dataframe)
