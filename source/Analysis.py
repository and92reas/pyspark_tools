from . import *



class Analysis:
    '''
    Containing functions that do not perform any transformations, but extract information from the available data
    '''

    def get_perc_of_null_values_for_feat(data):
        '''
        presents a list of the form (column name, percentage of nulls in the column) sorted by the percentage of nulls
        *data*: spark dataframe
        '''
        num_of_records = data.count()
        null_df = data.select([F.count(F.when((F.col(c).isNull()) | (F.isnan(F.col(c))) | (F.col(c) == None), c)).alias(c) for c in data.columns])
        columns = null_df.columns
        num_of_nulls = null_df.rdd.collect()[0]
        perc_of_nulls = [round((null_num/num_of_records), 2) for null_num in num_of_nulls]
        sorted_cols_null_perc = sorted(list(zip(columns, perc_of_nulls)), key = lambda x: x[1], reverse = True)
        return sorted_cols_null_perc

    def get_confidence_intervals_for_mean_using_bootstrap_percentiles_method(df,
                                                                         column,
                                                                         sample_perc = 0.1,
                                                                         samples = 50,
                                                                         confidence = 0.95):
        '''
        returns a tuple of the form (lower confidence boundary, mean, upper confidence boundary) for the values of a particular column of the dataframe
        *df*: spark dataframe
        *column*: the name of the column that you are interested in
        *sample_perc*: the percentage of the dataset acquired in each samPle
        *samples*: the number of samples taken during the boostrapping process
        *confidence*: value belonging in the (0,1) interval denoting the required confidence degree for the confidence intervals
        '''
        lower_percentile = (1 - confidence) / 2
        upper_percentile = 1 - (((1 - confidence) / 2))
        seeds = list(range(samples))
        samples = [df.select(column).sample(sample_perc, seed).rdd.map(lambda row: row[column]).collect() for seed in seeds]
        sample_means = [np.mean(sample) for sample in samples]
        sample_mean_confidence_bounds = np.quantile(sample_means, [lower_percentile, upper_percentile])
        return (sample_mean_confidence_bounds[0], np.mean(sample_means), sample_mean_confidence_bounds[1])


    def get_highly_correlated_features(corr_matrix,
                                        corr_upper_thres = 0.75,
                                        corr_lower_thres = -0.75):
        '''
        returns a dictionary of the form (feature -> list of tuples of the form (highly correlated feature, correlation))
        *corr_matrix*: pandas dataframe containing the correlation matrix
        *corr_upper_thres*: correlations above this threshold will be considered as high correlations
        *corr_lower_thres*: correlations below this threshold will be considered as lower correlations
        '''
        def get_highly_correlated_features_to_column(column, index, corr_matrix_indices_columns):
            '''
            return a list of tuples of the form (feature, correlation) containing the most correlated features to a particular column, sorted by the absolute value of the correlation
            *column*: the name of the column
            *index*: the index of the column
            *corr_matrix_indices_columns*: list of the form (index of column, name of column)
            '''
            corr_matrix_indices_to_columns = dict(corr_matrix_indices_columns)
            col_correlations = list(zip(list(range(len(corr_matrix_indices_to_columns))), list(corr_matrix[column])))
            desc_col_correlations = sorted(col_correlations, reverse = True, key = lambda x: x[1])
            asc_col_correlations = desc_col_correlations[::-1]
            pos_high_corr_cols = list(map(lambda index_corr: (corr_matrix_indices_to_columns[index_corr[0]], index_corr[1]), list(filter(lambda index_corr: index_corr[1] >= corr_upper_thres, desc_col_correlations))))
            neg_high_corr_cols = list(map(lambda index_corr: (corr_matrix_indices_to_columns[index_corr[0]], index_corr[1]), list(filter(lambda index_corr: index_corr[1] <= corr_lower_thres, asc_col_correlations))))
            all_corr_cols = list(filter(lambda col_corr: col_corr[0] != column, pos_high_corr_cols + neg_high_corr_cols))
            return sorted(all_corr_cols, reverse = True, key = lambda x: abs(x[1]))




        corr_matrix_columns = corr_matrix.columns
        corr_matrix_indices_columns = list(zip(list(range(len(corr_matrix_columns))), corr_matrix_columns))
        return list(map(lambda index_col: (index_col[1], get_highly_correlated_features_to_column(index_col[1], index_col[0], corr_matrix_indices_columns)), corr_matrix_indices_columns))


    def present_outlier_info_for_feature(data,
                            feature,
                            rel_error = 0.01,
                            k = 1.5):
        '''
        print outlier information on feature
        *data*: spark dataframe
        *feature*: the name of the feature of interest
        *rel_error*: the relative error used for the approxQuantiles method, determining the precision-computational time trade-off
        *k*: the value of k parameter used for the calculation of the tukey fences
        '''
        q1, median, q3 = data.approxQuantile(feature, [0.25, 0.5, 0.75], relativeError = rel_error)
        minimum, maximum = data.select(feature, F.col(feature).alias("{}_".format(feature))).agg({feature : "min", "{}_".format(feature): "max"}).\
                                                                                    rdd.map(lambda row: (row["min({})".format(feature)], row["max({}_)".format(feature)])).collect()[0]
        iqr = q3 - q1
        outliers_df = data.select(feature).filter((F.col(feature) < q1 - k * iqr) | (F.col(feature) > q3 + k * iqr))
        low_outliers_count = outliers_df.filter(F.col(feature) < q1 - k * iqr).count()
        high_outliers_count = outliers_df.filter(F.col(feature) > q3 + k * iqr).count()
        data_count = data.count()
        print("q1: {}".format(q1))
        print("median: {}".format(median))
        print("q3: {}".format(q3))
        print("Number of low outliers: {} ({:.2f}%)".format(low_outliers_count, 100 * (low_outliers_count / data_count)))
        print("Number of high outliers: {} ({:.2f}%)".format(high_outliers_count, 100 * (high_outliers_count / data_count)))
