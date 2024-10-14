
def lists_to_categorical(df, outside_column, categorical_key, condition_key, condition_value):
    unique_categories = set()
    # find distinct values in the categorical column
    for list_of_dictionaries in df[outside_column]:
        for dictionary in list_of_dictionaries:
            if dictionary[condition_key] == condition_value:
                unique_categories.add(dictionary[categorical_key])

    return unique_categories
'''
    # carry out one-hot-encoding on categorical column
    for category in unique_categories:
        df[category] = df[outside_column].apply(
            lambda list_of_dictionaries: 1 if any(
                ingredient[categorical_key] == category for ingredient in list_of_dictionaries
            ) else 0
        )

    return df
'''

def column_count_dicts_with_condition(df, column_name, key, value, new_column_name):

    def count_matching_dicts(list_of_dicts):
        number_of_matching_dicts = 0
        if isinstance(list_of_dicts, list):
            for dict in list_of_dicts:
                if dict[key] == value:
                    number_of_matching_dicts += 1

        return number_of_matching_dicts


    df[new_column_name] = df[column_name].apply(count_matching_dicts)

    return df
