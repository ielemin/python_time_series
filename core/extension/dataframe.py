import pandas as pd
import numpy as np

__author__ = 'ielemin'


def groupby_date(self: pd.DataFrame):
    """Group DataFrame rows by their date, preserving DatetimeIndex structure.

    :param self: a DataFrame
    :return: a groupby object on the DataFrame
    """
    return self.groupby(pd.DatetimeIndex(self.index.date))


# Register
pd.DataFrame.groupby_date = groupby_date


def hedge(self: pd.DataFrame):
    """Subtract the row average series to every columns of a DataFrame.

    Much faster than broadcasting the mean series to a full df and subtracting it.
    :param self: a DataFrame
    :return: a DataFrame
    """
    mean = self.mean(axis=1)

    def _hedge(series: pd.Series):
        return series - mean

    return self.apply(_hedge)


# Register
pd.DataFrame.hedge = hedge


def subtract_series(self: pd.DataFrame, other: pd.Series):
    """Subtract a given series to every column of a DataFrame.

    Much faster than broadcasting the series to a full df and subtracting it.
    :param self: a DataFrame
    :param other: a Series
    :return: a DataFrame
    """

    def _subtract(series: pd.Series):
        return series - other

    return self.apply(_subtract)


# Register
pd.DataFrame.subtract_series = subtract_series


def in_quantile(self: pd.DataFrame, level: int, sort_by_abs=False, ascending=True, complement=False):
    """Return whether the items belong in the top (or bottom) L (or N-L) of their row.

    :param self: a DataFrame
    :param level: an int, the L-th item of the sorted row values
    :param sort_by_abs: if True, the items are sorted by their absolute value
    :param ascending: if True, the bottom L items are selected
    :param complement: if True, the selection is inverted (e.g. Top L becomes Bottom N-L)
    :return:
    """

    def _quantile_value(series: pd.Series):
        return series.sort_values(ascending=ascending).iloc[level]

    used = self
    if sort_by_abs:
        used = used.abs()

    quantile_value = used.apply(_quantile_value, axis=1)

    # Larger than quantile(L) if Top L or Bottom N-L
    # Smaller than quantile(L) if Bottom L or Top N-L
    selection_sign = (2 * ascending - 1) * (2 * complement - 1)

    mask = subtract_series(used, quantile_value) * selection_sign >= 0

    return mask, quantile_value


pd.DataFrame.in_quantile = in_quantile


def dewma(self: pd.DataFrame, penalty: pd.Series, scale, halflife, adjust=True, min_periods=None):
    """Dynamic exponential weighted moving average.

    The larger the penalty, the faster the output updates.

    For every Series {s_i}, the output {u_i} is given by the iteration
        u_0 = 0, and
        u_(n+1) = (1-w_n) x u_n + w_n x s_n

    The trick is to rewrite this as compositions of cumsum/cumprod that are very fast to compute in pandas
        u_0 = 0, and
        u_(n+1) = cumprod(0...n, 1-w_i) x cumsum(0...n, s_i x w_i x cumprod(0...i, 1-w_j)^(-1))

    We call below
        propagator(n) the quantity cumprod(0...n, 1-w_i) that accumulates the penalty, and
        contributor(n) the quantity w_i x cumprod(0...i, 1-w_j)^(-1) that scales the current update s_i

    :param self: a DataFrame
    :param penalty: a Series representing a positive penalty for each time point
    :param scale: a penalty larger than this will update twice as fast
    :param halflife: a wrapper to the pd.ewma parameter
    :param min_periods: a wrapper to the pd.ewma parameter
    :param adjust: a wrapper to the pd.ewma parameter
    :return: the weighted DataFrame, the Series of the normalization coefficient, the Series of the weights used
    """

    if pd.Series(penalty < 0).any():
        print('[WARNING] Negative penalty floored at zero')
        penalty[penalty < 0] = 0

    if len(self) != len(penalty):
        print('[ERROR] length mismatch between DataFrame ({0}) and penalty ({1})'.format(len(self), len(penalty)))
        return None

    regular_alpha = 1 - np.exp(- 1 / halflife)  # TODO check pd.ewma convention
    alpha = regular_alpha * np.exp(- penalty / scale)

    propagator = pd.Series(1 - alpha).cumprod()
    contributor = pd.Series(alpha / propagator)
    normalization = propagator * contributor.cumsum()

    def _do(series: pd.Series):
        contribution = series * contributor
        accumulation = propagator * contribution.cumsum()
        if adjust:
            accumulation /= normalization
        return accumulation

    output = self.apply(_do)

    if min_periods is not None and min_periods > 0:
        output.iloc[:min_periods, :] = np.nan

    return output, normalization, alpha

pd.DataFrame.dewma = dewma
