import pandas as pd
from utils import *
import pyarrow

# Import data using pandas
df = pd.read_json("C:/Users/domci/PycharmProjects/ml-solvro-recruitment/data/cocktail_dataset.json")

# Drop columns that won't be used for clustering and create unnecessary noise
df.drop(["alcoholic", "instructions", "imageUrl", "createdAt", "updatedAt"], axis=1, inplace=True)

# Change categorical columns to category datatype
df = df.astype({
    "category": "category",
    "glass": "category"
})

# Encode the 'category' column to binary(0 or 1) column 'ordinaryDrink'
encode_one_category_from_strings(df, "category", "ordinaryDrink")

# Generalize glass type's that occurred less than 3 times
df = generalize_items_in_column(df, "glass", 3, "Other")

# Encode generalized 'glass' column
df = one_hot_encode_column(df, "glass")

# Exclude a binary(0 or 1) column IBA from 'tags' column
df = encode_one_category_from_lists(df, "tags", "IBA")

# Create a column that contains an information about the number of used ingredients in each cocktail
df["numIngredients"] = df["ingredients"].apply(len)

# Create a column that shows how many of the cocktails ingredients are alcoholic
df["numAlcoholicIngredients"] = df["ingredients"].apply(count_matching_dicts, args=("alcohol", 1))

# Encode most used ingredients in the ingredients column
sorted_items_counter = sorted_column_items_counter(df, "ingredients", "name")

for ingredient in list(sorted_items_counter.keys()) [:10]:
    df[ingredient] = df["ingredients"].apply(
        lambda ingredients_list: 1 if any(ingredient_dict["name"] == ingredient for ingredient_dict in ingredients_list) else 0
    )

# Change columns names to standardize them
df.columns = ['id', 'name', 'ingredients', 'ordinaryDrink',
      'glassChampagneFlute', 'glassCocktailGlass', 'glassCollinsGlass',
       'glassHighballGlass', 'glassOld-fashionedGlass', 'glassOther',
       'glassWhiskeySour', 'IBA', 'numIngredients', 'numAlcoholicIngredients', 'gin', 'lightRum',
        'tripleSec', 'sugar', 'lemonJuice', 'lemon', 'powderedSugar',
        'lemonPeel', 'lime', 'dryVermouth']

# Save the final dataframe to parquet so that it saves the information about columns datatypes
df.to_parquet("../data/final_cocktail_dataset.parquet")
