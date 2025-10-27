import pandas as pd
from collections import Counter

FBC_COLS = [
    "Basophils",
    "Eosinophils",
    "Haematocrit",
    "Haemoglobin",
    "Lymphocytes",
    "MCH",
    "MCV",
    "Monocytes",
    "Neutrophils",
    "RBC",
    "WBC",
]


def build_derivation_cols():
    derivation_cols = {
        "MCV": [
            ("Haematocrit", "RBC"),  # calculate_mcv
        ],
        "Haematocrit": [
            ("MCV", "RBC"),  # calculate_haematocrit_from_mcv_rbc
            ("Haemaglobin", "MCHC"),  # calulate_haematocrit_from_haem_mchc
        ],
        "RBC": [
            ("Haematocrit", "MCV"),  # calculate_rbc_from_haematocrit_mcv
            ("Haemoglobin", "MCH"),  # calculate_rbc_from_haem_mch
        ],
        "Haemoglobin": [
            ("RBC", "MCH"),  # calculate_haemoglobin_rbc_mch
            ("Haematocrit", "MCH"),  # calculate_haemoglobin_haematocrit_mch
        ],
        "MCH": [
            ("Haemoglobin", "RBC"),  # calculate_mch_haem_rbc
        ],
        "MCHC": [
            ("Haemoglobin", "Haematocrit")  # calculate_mchc_from_haem_haematocrit
        ],
        "WBC": [
            (
                "Basophils",
                "Eosinophils",
                "Lymphocytes",
                "Monocytes",
                "Neutrophils",
            )  # calculate_wbc
        ],
        "Basophils": [
            (
                "WBC",
                "Eosinophils",
                "Lymphocytes",
                "Monocytes",
                "Neutrophils",
            )  # calculate_missing_wbc
        ],
        "Eosinophils": [
            ("WBC", "Basophils", "Lymphocytes", "Monocytes", "Neutrophils")
        ],
        "Lymphocytes": [
            ("WBC", "Basophils", "Eosinophils", "Monocytes", "Neutrophils")
        ],
        "Monocytes": [
            ("WBC", "Basophils", "Eosinophils", "Lymphocytes", "Neutrophils")
        ],
        "Neutrophils": [
            ("WBC", "Basophils", "Eosinophils", "Lymphocytes", "Monocytes")
        ],
    }
    return derivation_cols


def calculate_mcv(haematocrit, rbc_count):
    """
    Haematocrit: (L/L)
    RBC_count: 10^12/L
    """
    return (haematocrit * 1000) / rbc_count


def calculate_haematocrit_from_mcv_rbc(mcv, rbc_count):
    """
    MCV: fL
    RBC_count: 10^12/L
    """
    return (mcv * rbc_count) / 1000


def calulate_haematocrit_from_haem_mchc(haemoglobin, mchc):
    """ "
    haemoglobin: g/dL
    mchc: g/dL
    """
    return haemoglobin / mchc


def calculate_rbc_from_haematocrit_mcv(haematocrit, mcv):
    return (haematocrit * 10) / mcv


def calculate_rbc_from_haem_mch(haemoglobin, mch):
    """
    mch: pg
    """
    return (10 * haemoglobin) / mch


def calculate_haemoglobin_rbc_mch(rbc, mch):
    return (rbc * mch) / 10


def calculate_haemoglobin_haematocrit_mch(haematocrit, mch):
    return haematocrit * mch


def calculate_mch_haem_rbc(haemoglobin, rbc):
    return haemoglobin / rbc


def calculate_mchc_from_haem_haematocrit(haemoglobin, haematocrit):
    return haemoglobin / haematocrit


def calculate_wbc(basophils, eosinophils, lymphocytes, monocytes, neutrophils):
    return basophils + eosinophils + lymphocytes + monocytes + neutrophils


def calculate_missing_wbc(wbc, *present_params):
    missing_param_count = sum(present_params)
    return wbc - missing_param_count


# Additional functions:

# def calculate_basophils_from_proportion(basophil_proportion, wbc):
#   return (basophil_proportion/100)*wbc
#
# def calculate_basophil_proportion(basophils, wbc):
#   return (basophil/wbc) * 100

# def calculate_eosinophil_from_proportion(eosinophil_proportion, wbc):
#   return (eosinophil_proportion/100)*wbc
#
# def calculate_eosinophil_proportion(eosinophil, wbc):
#   return (eosinophil/wbc) * 100


# def calculate_lymphocytes_from_proportion(lymphocytes_proportion, wbc):
#   return (lymphocytes_proportion/100)*wbc
#
# def calculate_lymphocytes_proportion(lymphocytes, wbc):
#   return (lymphocytes/wbc) * 100

# def calculate_monocytes_from_proportion(monocytes_proportion, wbc):
#   return (monocytes_proportion/100)*wbc
#
# def calculate_monocytes_proportion(monocytes, wbc):
#   return (monocytes/wbc) * 100

# def calculate_neutrophils_from_proportion(neutrophils_proportion, wbc):
#   return (neutrophils_proportion/100)*wbc
#
# def calculate_neutrophils_proportion(neutrophils, wbc):
#   return (neutrophils/wbc) * 100


def build_compute_funcs():
    compute_funcs = {
        "MCV": [calculate_mcv],
        "Haematocrit": [
            calculate_haematocrit_from_mcv_rbc,
            calulate_haematocrit_from_haem_mchc,
        ],
        "RBC": [calculate_rbc_from_haematocrit_mcv, calculate_rbc_from_haem_mch],
        "Haemoglobin": [
            calculate_haemoglobin_rbc_mch,
            calculate_haemoglobin_haematocrit_mch,
        ],
        "MCH": [calculate_mch_haem_rbc],
        "MCHC": [calculate_mchc_from_haem_haematocrit],
        "WBC": [calculate_wbc],
        "Basophils": [calculate_missing_wbc],
        "Eosinophils": [calculate_missing_wbc],
        "Lymphocytes": [calculate_missing_wbc],
        "Monocytes": [calculate_missing_wbc],
        "Neutrophils": [calculate_missing_wbc],
    }
    return compute_funcs


def derive_variable(col_to_derive, data, derivations, compute_funcs, source_df):
    data = data.copy()
    n_derivation_funcs = len(compute_funcs[col_to_derive])
    for i in range(n_derivation_funcs):
        func = compute_funcs[col_to_derive][i]
        input_cols = derivations[col_to_derive][i]

        if any(col not in data.columns for col in input_cols):
            continue

        mask_target_missing = data[col_to_derive].isna()
        mask_inputs_present = data[list(input_cols)].notna().all(axis=1)
        mask = mask_target_missing & mask_inputs_present

        data.loc[mask, col_to_derive] = func(
            *[data.loc[mask, col] for col in input_cols]
        )
        source_df.loc[mask, col_to_derive] = f"{func.__name__}"
    return data, source_df


def derive_missing_parameters(data):
    derivations = build_derivation_cols()
    compute_funcs = build_compute_funcs()
    source_df = pd.DataFrame(data=data.notna(), columns=FBC_COLS, dtype=str)
    max_iter = len(FBC_COLS)
    iters = 0

    derived_data = data.copy()
    while iters < max_iter:
        for variable in FBC_COLS:
            derived_data, source_df = derive_variable(
                variable, derived_data, derivations, compute_funcs, source_df
            )

        iters += 1

    print(data[FBC_COLS].isna().sum(axis=0) - derived_data[FBC_COLS].isna().sum(axis=0))
    return derived_data, source_df


def missingness_counts_fbc_rows(data):
    mask = data[FBC_COLS].notna().sum(axis=1) > 0
    missing_counts_per_column = data.loc[mask, FBC_COLS].isna().sum()
    print("Number of rows with an FBC blood test present:", len(data.loc[mask]))
    return missing_counts_per_column


def missingness_percentage_fbc_rows(data):
    mask = data[FBC_COLS].notna().sum(axis=1) > 0
    missing_counts_per_column = data.loc[mask, FBC_COLS].isna().sum()
    percentage_missing = round(100 * (missing_counts_per_column / len(data[mask])), 2)
    return percentage_missing


def count_sources(source_df):
    mapped_source_df = source_df.map(map_true_false)
    counter = Counter()
    for col in FBC_COLS:
        counter.update(
            {
                name: count
                for (i, (name, count)) in mapped_source_df[col]
                .value_counts()
                .reset_index()
                .iterrows()
            }
        )
    return counter


def map_true_false(val):
    if val == "True":
        return "Already present"
    if val == "False":
        return "Remains missing"
    return val
