import pandas as pd
from utils import *

# Import data using pandas
df = pd.read_json("../data/cocktail_dataset.json")

# Drop columns that won't be used for clustering and may create unnecessary noise
df.drop(["alcoholic", "instructions", "imageUrl", "createdAt", "updatedAt"], axis=1, inplace=True)

# Convert categorical columns to category datatype
df = df.astype({
    "category": "category",
    "glass": "category"
})

# Encode the 'category' column to binary(0 or 1) column 'ordinaryDrink'
df = encode_one_category_from_strings(df, "category", "ordinaryDrink")

# Generalize glass type's that occur less than 3 times
df = generalize_items_in_column(df, "glass", 3, "Other")

# One-hot encode generalized 'glass' column
df = one_hot_encode_column(df, "glass")

# Exclude a binary(0 or 1) column "IBA" from "tags" column
df = encode_one_category_from_lists(df, "tags", "IBA")

# Create a column that indicates the number of used ingredients in each cocktail
df["numIngredients"] = df["ingredients"].apply(len)

# Create a column that shows how many of the cocktails ingredients are alcoholic
df["numAlcoholicIngredients"] = df["ingredients"].apply(count_matching_dicts, args=("alcohol", 1))

# Encode the most used ingredients in the "ingredients" column
sorted_items_counter = sorted_column_items_counter(df, "ingredients", "name")

for ingredient in list(sorted_items_counter.keys()) [:20]:
    df[ingredient] = df["ingredients"].apply(
        lambda ingredients_list: 1 if any(
            ingredient_dict["name"] == ingredient
            for ingredient_dict in ingredients_list
        ) else 0
    )

# Drop the "ingredients" column as it is no longer needed
df.drop("ingredients", axis=1, inplace=True)

# Save the final dataframe to parquet so that it saves the information about columns datatypes
df.to_parquet("../data/final_cocktail_dataset.parquet")
