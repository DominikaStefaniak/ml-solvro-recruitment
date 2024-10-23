from unittest.mock import inplace

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_one_category_from_strings(df, column_name, category_name):
    df[category_name] = df[column_name].apply(lambda category: 1 if category == "Ordinary Drink" else 0)
    df.drop(column_name, axis=1, inplace=True)

    return df

def generalize_items_in_column(df, column_name, condition, generalized_name):
    distinct_values_count = df[column_name].value_counts()
    condition_indexes = []
    for index, value in distinct_values_count.items():
        if value < condition:
            condition_indexes.append(index)
    df[column_name] = df[column_name].apply(lambda item: generalized_name if item in condition_indexes else item)

    return df


def one_hot_encode_column(df, column_name):
    enc = OneHotEncoder(sparse_output=False)
    encoded_column = enc.fit_transform(df[[column_name]])
    encoded_column_df = pd.DataFrame(encoded_column, columns=enc.get_feature_names_out([column_name]))
    df = pd.concat([df, encoded_column_df], axis=1)
    df.drop(column_name, axis=1, inplace=True)

    return df


def encode_one_category_from_lists(df, column_name, category_name):
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])
    df[category_name] = df[column_name].apply(lambda x: 1 if category_name in x else 0)
    df.drop(column_name, axis=1, inplace=True)

    return df


def list_of_dictionaries_unique_categories(df, outside_column, categorical_key, condition_key, condition_value):
    unique_categories = set()
    for list_of_dictionaries in df[outside_column]:
        for dictionary in list_of_dictionaries:
            if dictionary[condition_key] == condition_value:
                unique_categories.add(dictionary[categorical_key])

    return unique_categories

def count_matching_dicts(list_of_dicts, key, value):
    number_of_matching_dicts = 0
    if isinstance(list_of_dicts, list):
        for dictionary in list_of_dicts:
            if dictionary[key] == value:
                number_of_matching_dicts += 1

    return number_of_matching_dicts