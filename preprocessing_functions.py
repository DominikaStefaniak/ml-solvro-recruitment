
def list_of_dictionaries_unique_categories(df, outside_column, categorical_key, condition_key, condition_value):
    unique_categories = set()
    # find distinct values in the categorical column
    for list_of_dictionaries in df[outside_column]:
        for dictionary in list_of_dictionaries:
            if dictionary[condition_key] == condition_value:
                unique_categories.add(dictionary[categorical_key])

    return unique_categories


def count_matching_dicts(list_of_dicts, key, value):
    number_of_matching_dicts = 0
    if isinstance(list_of_dicts, list):
        for dict in list_of_dicts:
            if dict[key] == value:
                number_of_matching_dicts += 1

    return number_of_matching_dicts
