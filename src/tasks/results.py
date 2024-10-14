import lib.util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns


def toxicity_barplot(df: pd.DataFrame, ax: matplotlib.axes.Axes):
    """
    Create a bar plot displaying the mean toxicity scores for different conversation variants, 
    grouped by annotator prompts.

    This function generates a horizontal bar plot where the x-axis represents toxicity 
    scores, and the y-axis represents different conversation variants. The bars are 
    colored by annotator demographic. An additional vertical red line is plotted at a 
    toxicity score of 3 to mark a threshold.

    :param df: The input DataFrame containing the toxicity scores, conversation variants, and annotator prompts.
    :type df: pd.DataFrame
    :param ax: The matplotlib axes object where the bar plot will be drawn.
    :type ax: matplotlib.axes.Axes
    :return: None

    :example:
        >>> fig, example_ax = plt.subplots()
        >>> toxicity_barplot(df, example_ax)
        >>> plt.show()
    """

    sns.barplot(
        data=df,
        y="conv_variant",
        x="toxicity",
        hue="annotator_prompt",
        estimator=np.mean,
        ax=ax,
    )
    ax.axvline(x=3, color="r")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim(0, 5)
    ax.legend(title="Annotator Demographic", fontsize="6", title_fontsize="6.5", loc="upper right")


def pvalue_heatmap(
    value_df: pd.DataFrame,
    pvalue_df: pd.DataFrame,
    show_labels: bool = False,
    correlation_title: str = "",
    xlabel_text: str = "",
    filename: str | None = None,
    output_dir: str = "."
) -> None:
    """
    Generate a heatmap visualizing correlation (or other) values along with p-value significance.

    This function produces a heatmap where the lower triangle of the matrix contains
    correlation values from `value_df`. These values are annotated with asterisks based
    on the significance levels of corresponding p-values from `pvalue_df`.
    The heatmap can be saved to a file if a filename is specified.

    :param value_df: DataFrame containing the correlation or other values to be visualized.
    :type value_df: pd.DataFrame
    :param pvalue_df: DataFrame containing p-values corresponding to the values in `value_df`.
    :type pvalue_df: pd.DataFrame
    :param show_labels: Whether to display axis labels on the heatmap, defaults to False.
    :type show_labels: bool, optional
    :param correlation_title: Title for the heatmap, defaults to an empty string.
    :type correlation_title: str, optional
    :param xlabel_text: Label for the x-axis of the heatmap, defaults to an empty string.
    :type xlabel_text: str, optional
    :param filename: Optional filename to save the heatmap image, defaults to None.
    :type filename: str | None, optional
    :param output_dir: Directory where the heatmap image will be saved if a filename is provided, defaults to
    the current directory.
    :type output_dir: str, optional
    :return: None

    :example:
        >>> pvalue_heatmap(value_df, pvalue_df, show_labels=True, correlation_title="Correlation Heatmap")
    """
    # Format the value_df with asterisks based on pvalue_df
    formatted_values = _format_with_asterisks(value_df, pvalue_df)

    # Define tick labels
    ticklabels = value_df.columns if show_labels else "auto"

    # Create the heatmap
    sns.heatmap(
        np.tril(value_df),
        annot=np.tril(formatted_values),
        fmt="",  # This allows us to use strings with asterisks
        cmap="icefire",
        mask=_upper_tri_masking(value_df),
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cbar_kws={"label": "Mean Toxicity Difference"},
        annot_kws={"fontsize":8}
    )

    plt.title(correlation_title)
    plt.xlabel(xlabel_text)

    # Save the plot if a filename is provided
    if filename is not None:
        lib.util.save_plot(filename=filename, dir_name=output_dir)

    # Show the plot
    plt.show()


# code from https://stackoverflow.com/questions/47314754/how-to-get-triangle-upper-matrix-without-the-diagonal-using-numpy
def _upper_tri_masking(array: np.array) -> np.array:
    """Generate a mask for the upper triangular of a NxN matrix, without the main diagonal

    :param array: the NxN matrix
    :type array: np.array
    :return: the mask
    :rtype: np.array
    """
    m = array.shape[0]
    r = np.arange(m)
    mask = r[:, None] <= r
    return mask


def _format_with_asterisks(
    value_df: pd.DataFrame, pvalue_df: pd.DataFrame
) -> pd.DataFrame:
    """Format the values in the value_df with asterisks based on p-value significance levels

    :param value_df: DataFrame containing the values to display
    :param pvalue_df: DataFrame containing the p-values
    :return: DataFrame with values formatted with asterisks
    """
    formatted_df = value_df.copy().astype(str)
    for i in range(value_df.shape[0]):
        for j in range(value_df.shape[1]):
            value = value_df.iloc[i, j]
            pvalue = pvalue_df.iloc[i, j]
            if pd.notnull(pvalue):  # Only apply formatting if pvalue is not NaN
                if pvalue < 0.001:
                    num_asterisks = 3
                elif pvalue < 0.01:
                    num_asterisks = 2
                elif pvalue < 0.05:
                    num_asterisks = 1
                else:
                    num_asterisks = 0
            else: #if NaN
                num_asterisks = 0

            formatted_df.iloc[i, j] = f"{value:.3f}{num_asterisks * '*'}"

    return formatted_df
