"""
fbc_imputer module
Provides utilities to derive and impute missing full blood count (FBC) parameters
from other related measurements using clinical relationships. Equations are taken
from Pradeep Virdee's work:

Virdee, P.S., Fuller, A., Jacobs, M. et al. Assessing data quality from the
Clinical Practice Research Datalink: a methodological approach applied to the full
blood count blood test. J Big Data 7, 96 (2020).
https://doi.org/10.1186/s40537-020-00375-w

Primary concepts
- FBC_COLS: ordered list of FBC parameters this module knows how to handle.
- Derivation recipes (build_derivation_cols): For each derivable parameter,
    lists of input column tuples that can be used to compute it.
- Compute functions (build_compute_funcs): Maps target parameters to one or more
    Python callables that implement the corresponding computation.
- Source tracking: A DataFrame of the same shape as the FBC columns is used to
    record whether a value was originally present or derived (and by which function).
Intended usage
1. Prepare a pandas.DataFrame `data` containing any subset of the columns in
     FBC_COLS; missing values should be represented by NaN.
2. Call derive_missing_parameters(data) to attempt to derive missing FBC
     parameters using available inputs. The function iteratively applies all
     supported derivations until no further progress is possible (bounded by
     number of parameters to avoid infinite loops).
3. Optionally use missingness_counts_fbc_rows and missingness_percentage_fbc_rows
     to summarise remaining missingness among rows that have at least one FBC
     measurement. Use count_sources to tabulate how values were filled.
Calculation functions and units
- calculate_mcv(haematocrit, rbc_count)
    - haematocrit: fraction (L/L), e.g., 0.45 for 45%
    - rbc_count: 10^12 cells per litre (10^12/L)
    - returns MCV in femtolitres (fL); MCV = (haematocrit * 1000) / RBC
- calculate_haematocrit_from_mcv_rbc(mcv, rbc_count)
    - mcv: fL, rbc_count: 10^12/L
    - returns haematocrit as fraction (L/L); haematocrit = (MCV * RBC) / 1000
- calculate_haematocrit_from_haem_mchc(haemoglobin, mchc)
    - haemoglobin and mchc share units (typically g/dL)
    - returns haematocrit as fraction (haemoglobin / mchc)
    - elementwise for array-like inputs; callers should handle zeros in MCHC
- calculate_rbc_from_haematocrit_mcv(haematocrit, mcv)
    - returns RBC in 10^12/L; uses RBC = (haematocrit * 10) / MCV
- calculate_rbc_from_haem_mch(haemoglobin, mch)
    - returns RBC in 10^12/L; uses RBC = (10 * haemoglobin) / MCH
- calculate_haemoglobin_rbc_mch(rbc, mch)
    - returns haemoglobin (g/dL); haemoglobin = (RBC * MCH) / 10
- calculate_haemoglobin_haematocrit_mch(haematocrit, mch)
    - returns haemoglobin (g/dL); uses haemoglobin = haematocrit * MCH
- calculate_mch_haem_rbc(haemoglobin, rbc)
    - returns MCH (pg); MCH = (haemoglobin * 10) / RBC
- calculate_mchc_from_haem_haematocrit(haemoglobin, haematocrit)
    - returns MCHC (g/dL); MCHC = haemoglobin / haematocrit
- calculate_wbc(basophils, eosinophils, lymphocytes, monocytes, neutrophils)
    - sums individual leukocyte counts; all in 10^9/L
- calculate_missing_wbc(wbc, *present_params)
    - computes a missing leukocyte subtype by subtracting the sum of present
        subtype counts from the total WBC; accepts scalar or array-like inputs
Key functions
- build_derivation_cols()
    - Returns the dictionary that enumerates which input tuples can derive each
        target parameter. The keys are column names and values are lists of input
        tuples in the same order as compute functions for that target.
- build_compute_funcs()
    - Returns a mapping of column names to list(s) of callables. The i-th callable
        corresponds to the i-th input tuple in build_derivation_cols() for that key.
- derive_variable(col_to_derive, data, derivations, compute_funcs, source_df)
    - Attempts to derive a single column using all available recipes in order.
    - Operates on a copy of `data` and updates `source_df` on rows where the
        target was missing and all required inputs are present.
    - Inputs may be scalars or pandas Series/array-like; functions are invoked
        with column-wise slices (pandas Series) when applied to masked rows.
    - Returns the updated data and source_df.
- derive_missing_parameters(data)
    - Orchestrates iterative derivation across all FBC_COLS. Creates an initial
        source_df where present cells are marked "True" and missing as "False"
        (strings), then records function names when values are derived.
    - Iterates up to len(FBC_COLS) times to allow chained derivations.
    - Returns (derived_data, source_df).
- missingness_counts_fbc_rows(data)
    - For rows with at least one FBC measurement, returns the count of missing
        values per FBC parameter and prints the number of such rows.
- missingness_percentage_fbc_rows(data)
    - For rows with at least one FBC measurement, returns the percentage missing
        per parameter (rounded to 2 decimals).
- count_sources(source_df)
    - Maps the "True"/"False" markers to human-readable labels and counts values
        across the source tracking DataFrame. Returns a collections.Counter of
        observed source labels.
- map_true_false(val)
    - Helper to convert "True" -> "Already present", "False" -> "Remains missing",
        or otherwise return the original value.
Behavioral notes and assumptions
- All derivations are deterministic algebraic transforms; no statistical
    imputation is performed.
- Missing data handling: derivations are applied only where the target is NaN
    and all required inputs are non-NaN. Array-like inputs preserve NaNs during
    elementwise operations consistent with pandas/NumPy semantics.
- Unit expectations: callers are responsible for providing compatible units. A
    mismatch (e.g., haemoglobin in g/L vs g/dL) will yield incorrect results.
- Division by zero is not explicitly trapped in all functions; callers should
    ensure inputs avoid invalid values (or handle resulting NaNs/Infs from pandas
    or NumPy operations).
- The module is written to be dataframe-agnostic but expects pandas.DataFrame.
Example (conceptual)
- Provide a DataFrame with any subset of FBC_COLS (some NaNs). Call
    derive_missing_parameters(df) to get a filled DataFrame plus a source_df that
    records whether each value was originally present or was derived and by which
    function.
"""

from collections import Counter
import pandas as pd


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
    """ "
    Defines which columns can be used to derive other columns.
    Each key is a column that can be derived, and the value is a list of tuples.
    Each tuple contains the columns required to derive the key column using a specific function.
    """
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
    Calculate the mean corpuscular volume (MCV).

    Parameters
    ----------
    haematocrit : float
        Haematocrit as a fraction (L/L). For example, 0.45 for 45%.
    rbc_count : float
        Red blood cell count in 10^12 cells per litre (10^12/L). For example, 5.0.

    Returns
    -------
    float
        Mean corpuscular volume in femtolitres (fL).

    Raises
    ------
    ZeroDivisionError
        If `rbc_count` is zero.
    ValueError
        If `haematocrit` or `rbc_count` is negative.

    Notes
    -----
    The function uses the relationship MCV (fL) = (haematocrit * 1000) / rbc_count,
    which is equivalent to the clinical formula MCV = (haematocrit (%) * 10) / RBC
    when haematocrit is expressed as a percentage.

    Examples
    --------
    >>> calculate_mcv(0.45, 5.0)
    90.0
    """
    return (haematocrit * 1000) / rbc_count


def calculate_haematocrit_from_mcv_rbc(mcv, rbc_count):
    """
    Calculate haematocrit (packed cell volume) from mean corpuscular volume (MCV) and red blood cell count (RBC).

    Parameters
    ----------
    mcv : float
        Mean corpuscular volume in femtolitres (fL).
    rbc_count : float
        Red blood cell count in 10^12 cells per litre (10^12/L).

    Returns
    -------
    float
        Haematocrit as a unitless fraction (L of red cells per L of blood). To express as a percentage, multiply the result by 100.

    Notes
    -----
    Uses the relation: haematocrit = (MCV * RBC_count) / 1000
    This converts MCV (fL) and RBC_count (10^12/L) into a unitless fraction.

    Example
    -------
    >>> calculate_haematocrit_from_mcv_rbc(85.0, 4.5)
    0.3825
    """
    return (mcv * rbc_count) / 1000


def calculate_haematocrit_from_haem_mchc(haemoglobin, mchc):
    """
    Calculate haematocrit from haemoglobin and MCHC.
    Parameters
    ----------
    haemoglobin : float, pandas.Series or numpy.ndarray
        Haemoglobin concentration. Must use the same units as `mchc` (for example g/dL).
    mchc : float, pandas.Series or numpy.ndarray
        Mean corpuscular haemoglobin concentration. Same units as `haemoglobin` (for example g/dL).
        Must not be zero.
    Returns
    float, pandas.Series or numpy.ndarray
        Calculated haematocrit as a unitless fraction (haemoglobin / mchc).
        If a percentage value is required, multiply the returned value by 100.
    Notes
    -----
    - The operation is elementwise for array-like inputs and preserves NaN values.
    - Division by zero will raise an error for scalars or produce inf/NaN for NumPy arrays;
      callers should validate or mask zero/invalid `mchc` values beforehand.
    """

    return haemoglobin / mchc


def calculate_rbc_from_haematocrit_mcv(haematocrit, mcv):
    """
    Calculate red blood cell count (RBC) from haematocrit and mean corpuscular volume (MCV).

    Parameters
    ----------
    haematocrit : float
        Haematocrit as a unitless fraction (L of red cells per L of blood).
    mcv : float
        Mean corpuscular volume in femtolitres (fL).

    Returns
    -------
    float
        Red blood cell count in 10^12 cells per litre (10^12/L).

    Notes
    -----
    The function uses the relationship RBC = (haematocrit * 10) / MCV.
    """
    return (haematocrit * 10) / mcv


def calculate_rbc_from_haem_mch(haemoglobin, mch):
    """
    Calculate red blood cell count (RBC) from haemoglobin and mean corpuscular haemoglobin (MCH).

    Parameters
    ----------
    haemoglobin : float
        Haemoglobin concentration in grams per decilitre (g/dL).
    mch : float
        Mean corpuscular haemoglobin in picograms (pg).

    Returns
    -------
    float
        Red blood cell count in 10^12 cells per litre (10^12/L).
    """
    return (10 * haemoglobin) / mch


def calculate_haemoglobin_rbc_mch(rbc, mch):
    """
    Calculate haemoglobin from red blood cell count (RBC) and mean corpuscular haemoglobin (MCH).

    Parameters
    ----------
    rbc : float
        Red blood cell count in 10^12 cells per litre (10^12/L).
    mch : float
        Mean corpuscular haemoglobin in picograms (pg).

    Returns
    -------
    float
        Haemoglobin concentration in grams per decilitre (g/dL).
    """
    return (rbc * mch) / 10


def calculate_haemoglobin_haematocrit_mch(haematocrit, mch):
    """
    Calculate haemoglobin from haematocrit and mean corpuscular haemoglobin (MCH).

    Parameters
    ----------
    haematocrit : float
        Haematocrit as a unitless fraction (L of red cells per L of blood).
    mch : float
        Mean corpuscular haemoglobin in picograms (pg).

    Returns
    -------
    float
        Haemoglobin concentration in grams per decilitre (g/dL).
    """
    return haematocrit * mch


def calculate_mch_haem_rbc(haemoglobin, rbc):
    """
    Calculate mean corpuscular haemoglobin (MCH) from haemoglobin and red blood cell count (RBC).

    Parameters
    ----------
    haemoglobin : float
        Haemoglobin concentration in grams per decilitre (g/dL).
    rbc : float
        Red blood cell count in 10^12 cells per litre (10^12/L).

    Returns
    -------
    float
        Mean corpuscular haemoglobin in picograms (pg).
    """
    return (haemoglobin * 10) / rbc


def calculate_mchc_from_haem_haematocrit(haemoglobin, haematocrit):
    """
    Calculate mean corpuscular haemoglobin concentration (MCHC) from haemoglobin and haematocrit.

    Parameters
    ----------
    haemoglobin : float
        Haemoglobin concentration in grams per decilitre (g/dL).
    haematocrit : float
        Haematocrit as a unitless fraction (L of red cells per L of blood).

    Returns
    -------
    float
        Mean corpuscular haemoglobin concentration in grams per decilitre (g/dL).
    """
    return haemoglobin / haematocrit


def calculate_wbc(basophils, eosinophils, lymphocytes, monocytes, neutrophils):
    """
    Calculate total white blood cell count (WBC) from individual leukocyte counts.

    Parameters
    ----------
    basophils : float
        Basophil count in 10^9 cells per litre (10^9/L).
    eosinophils : float
        Eosinophil count in 10^9 cells per litre (10^9/L).
    lymphocytes : float
        Lymphocyte count in 10^9 cells per litre (10^9/L).
    monocytes : float
        Monocyte count in 10^9 cells per litre (10^9/L).
    neutrophils : float
        Neutrophil count in 10^9 cells per litre (10^9/L).

    Returns
    -------
    float
        Total white blood cell count in 10^9 cells per litre (10^9/L).
    """
    return basophils + eosinophils + lymphocytes + monocytes + neutrophils


def calculate_missing_wbc(wbc, *present_params):
    """
    Calculate the missing white blood cell count (WBC) from the total WBC and the present leukocyte counts.
    """
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
    """
    Build a dictionary of computation functions for each FBC parameter.

    :return: A dictionary mapping FBC parameter names to their computation functions.
    :rtype: dict
    """
    compute_funcs = {
        "MCV": [calculate_mcv],
        "Haematocrit": [
            calculate_haematocrit_from_mcv_rbc,
            calculate_haematocrit_from_haem_mchc,
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
    """
    Derive a new variable in the dataset using the specified computation functions.

    :param col_to_derive: The name of the column to derive.
    :param data: The input DataFrame containing the data.
    :param derivations: A dictionary mapping column names to their derivation functions.
    :param compute_funcs: A dictionary mapping column names to their computation functions.
    :param source_df: A DataFrame to track the source of derived values.
    :return: The updated DataFrame and source DataFrame.
    """
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
    """
    Derive missing FBC parameters using available data.
    :param data: The input DataFrame containing FBC data.
    :return: The DataFrame with derived FBC parameters and the source DataFrame.
    """
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
    """
    Count the number of missing values for each FBC parameter in rows where at least one parameter is present.
    :param data: The input DataFrame containing FBC data.
    :return: A Series with the count of missing values for each FBC parameter.
    """
    mask = data[FBC_COLS].notna().sum(axis=1) > 0
    missing_counts_per_column = data.loc[mask, FBC_COLS].isna().sum()
    print("Number of rows with an FBC blood test present:", len(data.loc[mask]))
    return missing_counts_per_column


def missingness_percentage_fbc_rows(data):
    """
    Calculate the percentage of missing values for each FBC parameter in rows where at least one parameter is present.
    :param data: The input DataFrame containing FBC data.
    :return: A Series with the percentage of missing values for each FBC parameter.
    """
    mask = data[FBC_COLS].notna().sum(axis=1) > 0
    missing_counts_per_column = data.loc[mask, FBC_COLS].isna().sum()
    percentage_missing = round(100 * (missing_counts_per_column / len(data[mask])), 2)
    return percentage_missing


def count_sources(source_df):
    """
    Count the sources of derived values for each FBC parameter.
    :param source_df: The DataFrame tracking the source of derived values.
    :return: A Counter object with the count of sources for each FBC parameter.
    """
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
    """
    Map boolean string values to more descriptive labels.
    :param val: The input value to map.
    :return: The mapped value.
    """
    if val == "True":
        return "Already present"
    if val == "False":
        return "Remains missing"
    return val
