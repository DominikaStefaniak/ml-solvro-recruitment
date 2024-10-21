

def generalise_items_in_column(df, column_name, condition, generalised_name):
    distinct_values_count = df[column_name].value_counts()
    condition_indexes = []
    for index, value in distinct_values_count.items():
        if value < condition:
            condition_indexes.append(index)
    df[column_name] = df[column_name].apply(lambda item: generalised_name if item in condition_indexes else item)

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