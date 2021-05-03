from . import *

class Processing:

    '''
    contains functions used for making certain data trasformations
    '''

    def replace_outliers_with_percentile(data,
                       outlier_columns,
                       percentile_diff = 0.005,
                       rel_error = 0.01
                       ):
        '''
        replaces the outliers with the 0+diff and 100-diff percentile (for very small or large values respectively) and returns a tuple of the form (updated spark dataframe, dictionary of the form (outlier_column --> [low_percentile, high_percentile]))
        *data*: spark dataframe containing the dataset
        *outlier_columns*: list containing the numeric columns that are expected to have outliers
        *percentile_diff*: (percentile_diff) and (1- percentile_diff) percentiles will be used for the replacements
        *rel_error*: the relative error used by approxQuantile method
        '''
        def replace_outlier_for_col(data, col, low_perc, high_perc):
            '''
            replaces the outliers using the given percentiles and returns the updated datatet
            *data*: spark dataframe
            *low_perc*: the low percentile to replace the low outliers
            *high_perc*: the high percentile to replace the high outliers
            '''
            return data.withColumn(col, F.when(F.col(col) < low_perc, low_perc).otherwise(F.when(F.col(col) > high_perc, high_perc).otherwise(F.col(col))))

        cols_percentiles = [(col, list(data.approxQuantile(col, [percentile_diff, 1 - percentile_diff], relativeError = rel_error))) for col in outlier_columns]
        data = reduce(lambda data, col_perc: replace_outlier_for_col(data, col_perc[0], col_perc[1][0], col_perc[1][1]), cols_percentiles, data)
        return data, cols_percentiles


    def imput_null_values_with_mean_with_regards_to_other_column(data,
                                                            null_cols,
                                                            imput_col):
    '''
    replaces the null values of a list of columns with the column's mean with regards to a particular value of a reference column
    *data*: spark dataframe
    *null_col*: list of the columns whose null values should be imputted
    *imput_col*: the name of the column whose values will be used as points of reference for taking the imputations' means
    '''
    mean_per_segment = data.groupBy(imput_col).mean(*null_cols)
    data_with_cde_group_mean = data.join(mean_per_segment, imput_col)
    data = reduce(lambda data, col: data.withColumn(col, F.coalesce(col, "avg({})".format(col))).drop("avg({})".format(col)), null_cols, data_with_cde_group_mean)

    return data
