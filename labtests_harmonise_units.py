def correct_haemoglobin_values(df, cutoff):
    """Correct haemoglobin values in g/L to g/dL if they exceed the cutoff.
    Args:
        df (pd.DataFrame): DataFrame containing haemoglobin values.
        cutoff (float): Cutoff value to identify incorrect units.
        Returns:
        pd.DataFrame: DataFrame with corrected haemoglobin values.
    """
    df.loc[df["haemoglobin"] > cutoff, "haemoglobin"] /= 10
    return df


def correct_haematocrit_values(df, cutoff):
    """Correct haematocrit values if they exceed the cutoff.
    Args:
        df (pd.DataFrame): DataFrame containing haematocrit values.
        cutoff (float): Cutoff value to identify incorrect units.
        Returns:
        pd.DataFrame: DataFrame with corrected haematocrit values.
    """

    df.loc[df["haematocrit"] > cutoff, "haematocrit"] /= 10
    return df


function_dict = {
    "haemoglobin": correct_haemoglobin_values,
    "haematocrit": correct_haematocrit_values,
}


def correct_units(lab_test_df, cutoffs):
    """Correct lab test values based on provided cutoffs.
    Args:
        lab_test_df (pd.DataFrame): DataFrame containing lab test values.
        cutoffs (dict): Dictionary with column names as keys and cutoff values as values.
        Returns:
        pd.DataFrame: DataFrame with corrected lab test values.
    """
    for col, cutoff in cutoffs.items():
        if col in function_dict:
            lab_test_df = function_dict[col](lab_test_df, cutoff)
    return lab_test_df
