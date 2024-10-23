import pandas as pd
from utils import *

# Import data using pandas
dataframe = pd.read_json("C:/Users/domci/PycharmProjects/ml-solvro-recruitment/data/cocktail_dataset.json")

# Drop columns that won't be used for clustering and create unnecesary noise
dataframe.drop(["instructions", "imageUrl", "createdAt", "updatedAt"], axis=1, inplace=True)

# Change categorical columns to category datatype
dataframe = dataframe.astype({
    "category": "category",
    "glass": "category"
})

# Encode the 'category' column to binary column 'ordinaryDrink'
encode_one_category_from_strings(dataframe, "category", "ordinaryDrink")

# Generalize glass type's that occurred less than 3 times
dataframe = generalize_items_in_column(dataframe, "glass", 3, "Other")

# Encode generalized 'glass' column
dataframe = one_hot_encode_column(dataframe, "glass")

# Exclude a binary column IBA from 'tags' column
dataframe = encode_one_category_from_lists(dataframe, "tags", "IBA")

#dataframe["alcoholic_ingredients"] = dataframe["ingredients"].apply(count_matching_dicts, args=("alcohol", 0))

#dataframe['num_ingredients'] = dataframe['ingredients'].apply(len)

print(dataframe["IBA"])
