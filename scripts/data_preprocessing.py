import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import *
import pyarrow

# Import data using pandas
dataframe = pd.read_json("C:/Users/domci/PycharmProjects/ml-solvro-recruitment/data/cocktail_dataset.json")

# Drop columns that won't be used for clustering and create unnecessary noise
dataframe.drop(["alcoholic", "instructions", "imageUrl", "createdAt", "updatedAt"], axis=1, inplace=True)

# Change categorical columns to category datatype
dataframe = dataframe.astype({
    "category": "category",
    "glass": "category"
})

# Encode the 'category' column to binary(0 or 1) column 'ordinaryDrink'
encode_one_category_from_strings(dataframe, "category", "ordinaryDrink")

# Generalize glass type's that occurred less than 3 times
dataframe = generalize_items_in_column(dataframe, "glass", 3, "Other")

# Encode generalized 'glass' column
dataframe = one_hot_encode_column(dataframe, "glass")

# Exclude a binary(0 or 1) column IBA from 'tags' column
dataframe = encode_one_category_from_lists(dataframe, "tags", "IBA")

# Create a column that contains an information about the number of used ingredients in each cocktail
dataframe["numIngredients"] = dataframe["ingredients"].apply(len)

# Create a column that shows how may of the cocktails ingredients are alcoholic
dataframe["numAlcoholicIngredients"] = dataframe["ingredients"].apply(count_matching_dicts, args=("alcohol", 1))

#
sorted_items_counter = sorted_column_items_counter(dataframe, "ingredients", "name")

for ingredient in list(sorted_items_counter.keys()) [:10]:
    dataframe[ingredient] = dataframe["ingredients"].apply(
        lambda ingredients_list: 1 if any(ingredient_dict["name"] == ingredient for ingredient_dict in ingredients_list) else 0
    )

dataframe.columns = ['id', 'name', 'ingredients', 'ordinaryDrink',
      'glassChampagneFlute', 'glassCocktailGlass', 'glassCollinsGlass',
       'glassHighballGlass', 'glassOld-fashionedGlass', 'glassOther',
       'glassWhiskeySour', 'IBA', 'numIngredients', 'numAlcoholicIngredients', 'gin', 'lightRum',
        'tripleSec', 'sugar', 'lemonJuice', 'lemon', 'powderedSugar',
        'lemonPeel', 'lime', 'dryVermouth']

columns_to_convert = ["ordinaryDrink", 'glassChampagneFlute', 'glassCocktailGlass', 'glassCollinsGlass',
'glassHighballGlass', 'glassOld-fashionedGlass', 'glassOther', 'glass_WhiskeySour', 'IBA', 'gin',
'lightRum', 'tripleSec', 'sugar', 'lemonJuice', 'lemon', 'powderedSugar', 'lemonPeel', 'lime',
'dryVermouth']

dataframe[columns_to_convert] = dataframe[columns_to_convert].astype(bool)
# Scale the only not binary(0 or 1) column
#scaler = StandardScaler
#dataframe["numIngredients"] = scaler.fit_transform(dataframe["numIngredients"])

# Save the final dataframe to parquet so that it saves the information about columns datatypes
dataframe.to_parquet("../data/final_cocktail_dataset.parquet")