from . import *

class Plotting:

    '''
    contains functions that are used to generate plots (on a zeppelin notebook)
    '''


    def plot_density_plot(l,
                          x_min,
                          x_max,
                          y_min,
                          y_max,
                          x_label,
                          y_label,
                          bins,
                          confidence_intervals = False,
                          x_ci_lower_bound = None,
                          x_mean = None,
                          x_ci_upper_bound = None):
        '''
        producing a kernel density plot adding a vertical line one standard deviation away from the mean
        *l*: list containing the values to be plotted
        *x_min*: the minimum value of the x axis
        *x_max*: the maximum value of the x axis
        *y_min*: the minimum value of the y axis
        *y_max*: the maximum value of the y axis
        *x_label*: the label of the x axis
        *y_label*: the label of the y axis
        *bins*: the number of bins to be used
        *confidence_intervals*: boolean denoting whether we wish the confidence intervals of the mean to be added to the plot
        *x_ci_lower_bound*: the lower bound of the confidence interval
        *x_mean*: the mean of the bootstrapping samples
        *x_ci_upper_bound*: the lower bound of the confidence interval
        '''

        plt.clf()
        sns.distplot(l, kde=True, norm_hist=True, bins = bins)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if confidence_intervals:
            plt.vlines(x_ci_lower_bound, y_min, y_max, colors = 'red')
            plt.vlines(x_mean, y_min, y_max, colors = 'blue')
            plt.vlines(x_ci_upper_bound, y_min, y_max, colors = 'red')

        show(plt)


    def get_symmary_stats_and_density_plot(data,
                                           column,
                                           produce_density_plot = False,
                                           x_min = -999999999,
                                           x_max = 9999999999,
                                           y_min = 0,
                                           y_max = 1.1,
                                           bins = 10,
                                           rel_error = 0.01,
                                           confidence = None,
                                           bootstrapping_samples = 50,
                                           bootstrapping_sample_perc = 0.1):
        '''
        prints summary stats and optionally plots the density plot for a specific numeric feature
        *data*: spark dataframe containing the dataset
        *column*: the name of the column of interest
        *produce_density_plot*: boolean denoting whether the density plot should be plotted
        *x_min*: the minimum value plotted for the x axis
        *x_max*: the maximum value plotted for the x axis
        *y_min*: the minimum value plotted for the y axis
        *y_max*: the maximum value plotted for the y axis
        *bins*: the number of bins to be used for the distribution plot
        *rel_error*: the relative error used for the approxQuantiles function
        *confidence*: if not null, the confidence intervals will be plotted and this parameter will hold the degree of confidence required
        *bootstrapping_samples*: in case that the confidence intervals are plotted, this parameter holds the number of the samples taken by the bootstrapping procedure
        *bootstrapping_sample_perc*: in case the confidence intervals are plotted, this parameter holds the percentage of the values of each column
        '''
        def print_the_column_stats(count,
                                    mean,
                                    stdev,
                                    minimum,
                                    maximum,
                                    quartiles):
            '''
            it prints the column statistics
            *count*: the number of the valid values of the column
            *mean*: the mean value of the column
            *stdev*: the standard deviation of the column
            *minimum*: the minimum value of the column
            *maximum*: the maximum value of the column
            *quartiles*: list containing the 1st 2nd and 3rd quartile
            '''
            print("count:\t{}".format(count))
            print("mean:\t{:.2f}".format(float(mean)))
            print("std:\t{:.2f}".format(float(stdev)))
            print("min:\t{:.2f}".format(float(minimum)))
            print("q1:\t{:.2f}".format(float(quartiles[0])))
            print("median:\t{:.2f}".format(float(quartiles[1])))
            print("q3:\t{:.2f}".format(float(quartiles[2])))
            print("max:\t{:.2f}".format(float(maximum)))

        from .Analysis import Analysis

        x_label = column
        y_label = "probability"
        stats_dict = dict(data.select(column).describe().rdd.map(lambda row: (row["summary"], row[column])).collect())
        quartiles = data.approxQuantile(column, [0.25, 0.5, 0.75], relativeError = rel_error)
        print_the_column_stats(stats_dict["count"], stats_dict["mean"], stats_dict["stddev"], stats_dict["min"], stats_dict["max"], quartiles)
        if produce_density_plot:
            col_values = data.select(column).rdd.map(lambda row: row[column]).collect()
            if confidence is not None:
                low_bound, mean, upper_bound = Analysis.get_confidence_intervals_for_mean_using_bootstrap_percentiles_method(data, column, sample_perc = bootstrapping_sample_perc, samples = bootstrapping_samples, confidence = confidence)
                plot_density_plot(col_values, max(x_min, float(stats_dict["min"])), min(x_max, float(stats_dict["max"])), y_min, y_max, x_label, y_label, bins, confidence is not None, low_bound, mean, upper_bound)
            else:
                plot_density_plot(col_values, max(x_min, float(stats_dict["min"])), min(x_max, float(stats_dict["max"])), y_min, y_max, x_label, y_label, bins)



    def generate_heatmap_from_corr_matrix(df,
                                        feature_names,
                                        correlation_type = "spearman",
                                        heatmap_bottom = 0.25,
                                        heatmap_left = 0.1,
                                        heatmap_width = 1500,
                                        heatmap_height = 600,
                                        heatmap_linewidth = 0.3,
                                        heatmap_size = (10, 10)):
        '''
        generates the correlation matrix and plots it in the form of a heatmap
        *df: spark dataframe
        *feature_names*: list containing the names of the numerical features used for the creation of the correlation matrix
        *correlation_type*: the type of correlation to be used ('pearson' or 'spearman')
        *heatmap_bottom*: controlling the available space for the xlabelsticks
        *heatmap_left*: controlling the available space for the ylabelsticks
        *heatmap_width*: controlling the position of the heatmap in terms of the width
        *heatmap_heigh*t: controlling the position of the heatmap in terms of the height
        *heatmap_linewidth*: the required linewidth of the plot of the heatmap
        *heatmap_size*: setting the figsize parameter, controlling the size of the heatmap
        '''
        numerical_features_assembler = VectorAssembler(inputCols = feature_names, outputCol= "numerical_features")
        trans_df = numerical_features_assembler.transform(df)

        matrix = Correlation.corr(trans_df, "numerical_features", correlation_type).collect()[0]["{}({})".format(correlation_type, "numerical_features")].values
        correlation_matrix = np.reshape(matrix, (int(np.sqrt(len(matrix))), int(np.sqrt(len(matrix)))))
        correlation_df = pd.DataFrame(correlation_matrix, index = feature_names, columns = feature_names)

        plt.clf()
        plt.figure(figsize = heatmap_size)
        plt.gcf().subplots_adjust(bottom = heatmap_bottom, left = heatmap_left)
        sns.heatmap(correlation_df, xticklabels = 1, yticklabels = 1, linewidths = heatmap_linewidth, square = True)
        show(plt)
