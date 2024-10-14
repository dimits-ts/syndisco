import pandas as pd
import numpy as np
import scipy.stats
import scikit_posthocs
import diptest

from itertools import combinations


def posthoc_dunn(df: pd.DataFrame, val_col: str, group_col: str) -> pd.DataFrame:
    """
    Perform Dunn's post-hoc test with Bonferroni correction for multiple comparisons.
    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param val_col: The column name containing the values to be compared between groups.
    :type val_col: str
    :param group_col: The column name containing the group labels.
    :type group_col: str
    :return: A DataFrame containing pairwise p-values after Bonferroni correction.
    :rtype: pd.DataFrame

    :example:
        >>> example_df = pd.DataFrame({
        ...     'value': [1.2, 2.3, 2.5, 3.1],
        ...     'group': ['A', 'B', 'A', 'B']
        ... })
        >>> posthoc_dunn(example_df, val_col='value', group_col='group')
    """
    posthoc = scikit_posthocs.posthoc_dunn(
        df, val_col=val_col, group_col=group_col, p_adjust="bonferroni"
    )
    posthoc_df = posthoc.reset_index().melt(
        id_vars="index", var_name="Comparison", value_name="p-value"
    )
    posthoc_df.columns = ["Group1", "Group2", "p-value"]
    posthoc_df = posthoc_df.pivot(index="Group1", columns="Group2", values="p-value")
    return posthoc_df


# produced by ChatGPT
def pairwise_diffs(
    df: pd.DataFrame, groupby_cols: list[str], value_col: str
) -> pd.DataFrame:
    """
    Calculate pairwise differences in mean values between groups.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param groupby_cols: The columns to group by in order to calculate mean values.
    :type groupby_cols: list[str]
    :param value_col: The column name containing the values for which pairwise differences will be calculated.
    :type value_col: str
    :return: A DataFrame containing pairwise mean differences between groups.
    :rtype: pd.DataFrame

    :example:
        >>> example_df = pd.DataFrame({
        ...     'group': ['A', 'B', 'A', 'B'],
        ...     'value': [1.2, 2.3, 2.5, 3.1]
        ... })
        >>> pairwise_diffs(example_df, groupby_cols=['group'], value_col='value')
    """
    # calculate the mean toxicity for each combination of annotator_prompt and conv_variant
    mean_values = df.groupby(groupby_cols)[value_col].mean().unstack()

    # create an NxN DataFrame to store the pairwise differences
    annotator_prompts = mean_values.columns
    diff_matrix = pd.DataFrame(
        index=annotator_prompts, columns=annotator_prompts, dtype=float
    )

    # calculate pairwise differences between annotator prompts
    for annotator_prompt_1, annotator_prompt_2 in combinations(annotator_prompts, 2):
        differences = mean_values[annotator_prompt_1] - mean_values[annotator_prompt_2]
        average_diff = differences.mean()

        # Populate the difference matrix symmetrically
        diff_matrix.loc[annotator_prompt_1, annotator_prompt_2] = average_diff
        diff_matrix.loc[annotator_prompt_2, annotator_prompt_1] = -average_diff

    # no difference with itself
    np.fill_diagonal(diff_matrix.values, 0)

    return diff_matrix


def aposteriori_unimodality(grouped_annotations: list[np.array]) -> tuple[float, float]:
    """Run a statistical test for the aposteriori unimodality for annotations divided by a certain feature.
    If global nDFU > 0 and the retuned pvalue is low, then we reject the hypothesis that the
    feature does not explain the observed polarization.

    This test calculates the nDFU of all the annotations, and then the nDFU of each factor of the selected feature.
    If global nDFU > 0 but for all factors, nDFU_{factor} == 0, then we reject the aposteriori unimodality hypothesis.
    Instead of returning the individual nDFUs, this function runs a Wilcoxon test to determine if all nDFUs are 0.
    We use a non-parametric test because annotations rarely follow the normal distribution, and are typically few in number.

    :param grouped_annotations: the annotations, grouped by each factor for the selected feature
    :type grouped_annotations: list[np.array]
    :return: the global nDFU, and the 1-pvalue that all ndfus are zero
    :rtype: tuple[float, float]
    """
    # combine into flat array
    global_annotations = np.concatenate(grouped_annotations, axis=0)
    global_ndfu = ndfu(global_annotations)

    grouped_unimodality_pvalue = _groups_are_unimodal(grouped_annotations, global_ndfu)
    return global_ndfu, 1 - grouped_unimodality_pvalue


def _groups_are_unimodal(grouped_annotations: list[np.array], global_ndfu) -> float:
    """Test whether the nDFU of each factor of the feature are zero, using a Wilcoxon test.

    :param grouped_annotations: the annotations, grouped by each factor for the selected feature
    :type grouped_annotations: list[np.array]
    :return: the pvalue that all ndfus are zero
    :rtype: float
    """
    grouped_ndfu = np.array(
        [ndfu(annotation_group) for annotation_group in grouped_annotations]
    )
    # Use wilcoxon because we can not assume normality or random sampling
    # (realistically, annotations will be few)
    _, pvalue = scipy.stats.wilcoxon(
        grouped_ndfu, np.full_like(grouped_ndfu, fill_value=global_ndfu)
    )
    # H_0: all ndfus are 0 => feature explains polarization
    # H_a: at least 1 ndfu > 0 => feature does not explain polarization
    # use 1-pvalue to reverse the above. Now, if p is low, we accept that feature explains polarization
    return pvalue


# code from John Pavlopoulos https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def ndfu(input_data, histogram_input=True, normalised=True):
    """The normalized Distance From Unimodality measure
    :param: input_data: the data, by default the relative frequencies of ratings
    :param: histogram_input: False to compute rel. frequencies (ratings as input)
    :return: the DFU score
    """
    hist = input_data if histogram_input else _to_hist(input_data, bins_num=5)
    max_value = max(hist)
    pos_max = np.where(hist == max_value)[0][0]
    # right search
    max_diff = 0
    for i in range(pos_max, len(hist) - 1):
        diff = hist[i + 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    for i in range(pos_max, 0, -1):
        diff = hist[i - 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    if normalised:
        return max_diff / max_value
    return max_diff


def _to_hist(scores, bins_num=3, normed=True):
    """Creating a normalised histogram
    :param: scores: the ratings (not necessarily discrete)
    :param: bins_num: the number of bins to create
    :param: normed: whether to normalise or not, by default true
    :return: the histogram
    """
    # not keeping the values order when bins are not created
    counts, bins = np.histogram(a=scores, bins=bins_num)
    counts_normed = counts / counts.sum()
    return counts_normed if normed else counts
