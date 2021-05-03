from . import *

class Modelling:

    '''
    Contains functions responsible for the preparation of data for the modelling stage, the modelling stage and for extracting the feature importance of the models used for the modelling stage
    '''

    def set_up_pipeline(categorical_predictors = [],
                    numerical_predictors = [],
                    binary_predictors = [],
                    standardization = True):
    '''
    sets up and returns pyspark pipeline
    *categorical_predictors*: list containing the names of the categorical predictors
    *numerical_predictors*: list containing the names of the numerical predictors
    *binary_predictors*: list containing the names of the binary predictors
    *standardization*: boolean flag denoting whether the nunmerical features should be standardized or normalized
    '''

    indexed_categorical_predictors = list(map(lambda colname: "indexed_{}".format(colname), categorical_predictors))

    one_hot_categorical_predictors = list(map(lambda colname: "one_hot_{}".format(colname), categorical_predictors))

    indexers = [StringIndexer(inputCol=categorical_predictors[i], outputCol= indexed_categorical_predictors[i]) for i in range(len(categorical_predictors))]

    numerical_predictors_to_scale_assembler = VectorAssembler(inputCols = numerical_predictors, outputCol= "numerical_predictors_to_scale")

    if standardization:
        scaler = StandardScaler(inputCol="numerical_predictors_to_scale", outputCol="scaled_numerical_predictors", withStd=True, withMean=True)
    else:
        scaler = MinMaxScaler(inputCol="numerical_predictors_to_scale", outputCol="scaled_numerical_predictors")

    binary_numerical_predictors_assembler = VectorAssembler(inputCols = binary_predictors, outputCol="binary_numerical_predictors")

    encoder = OneHotEncoderEstimator(inputCols = indexed_categorical_predictors, outputCols = one_hot_categorical_predictors, dropLast = True)

    final_assembler = VectorAssembler(
        inputCols = ["scaled_numerical_predictors", "binary_numerical_predictors"] + one_hot_categorical_predictors,
        outputCol= "features")


    pipeline = Pipeline(
        stages =
        indexers +
        [
        numerical_predictors_to_scale_assembler,
        scaler,
        binary_numerical_predictors_assembler,
        encoder,
        final_assembler
        ]


    )

    return pipeline


    def splitting_data_and_applying_pipeline(data,
                                             pipeline,
                                             train_test_ratio = [0.8, 0.2],
                                             seed = 2021,
                                             label = "label"
                                             ):
        '''
        splits the data to training and test set and applies the pipeline transformations to both the training and the test set. Finally, a tuple of the form (spark dataframe containing the training set, spark dataframe containing the test set are returned, the fitted pipeline)
        *data*: spark dataframe containing the whole dataset
        *pipeline*: spark ml pipeline
        *train_test_ratio*: list tuple of the form (percentage of records forming the training set, percentage of records forming the test set)
        *label*: the name of the column containing the label
        '''
        train_data, test_data = data.randomSplit(train_test_ratio, seed = seed)

        fitted_pipeline = pipeline.fit(train_data)

        train_data = fitted_pipeline.transform(train_data)
        train_data.cache()

        test_data = fitted_pipeline.transform(test_data)

        return train_data, test_data, fitted_pipeline


    def training_and_evaluating(model,
                                train_data,
                                test_data,
                                model_identifier,
                                prob_thres = 0.5):
        '''
        trains the model, acquires the predictions and returns a tuple of the form (spark dataframe containing the predictions of the model, trained spark.ml model instance)
        *model*: spark.ml model
        *train_data*: spark dataframe containing the training set
        *test_data*: spark dataframe containig the test set
        *model_identifier*: the identifier of the model
        *prob_thres*: the probability threshold over which the prediction will be 1
        '''

        from .PythonUdf import PythonUdf
        from .Evaluation import Evaluation

        trained_model = model.fit(train_data)
        predictions = trained_model.transform(test_data)

        if prob_thres != 0.5:
            predictions = predictions.withColumn("processed_prediction", F.when(PythonUdf.sparse_to_array(F.col("probability"))[1] >= prob_thres , 1).otherwise(0))
        else:
            predictions = predictions.withColumn("processed_prediction", F.col("prediction"))
        predictions = predictions.withColumn("labels_predictions", F.concat(F.col("label"), F.lit("_"), F.col("processed_prediction").cast(IntegerType())))
        labels_predictions = predictions.select("labels_predictions").toPandas()
        labels_series = pd.Series(labels_predictions["labels_predictions"].str.split("_").str.get(0))
        preds_series = pd.Series(labels_predictions["labels_predictions"].str.split("_").str.get(1))

        model_areaUnderROC = BinaryClassificationEvaluator(metricName = "areaUnderROC")
        model_areaUnderPR = BinaryClassificationEvaluator(metricName = "areaUnderPR")
        model_precision = Evaluation.get_precision_of_label('1', labels_series, preds_series)
        model_recall = Evaluation.get_recall_of_label('1', labels_series, preds_series)
        model_accuracy = Evaluation.get_multiclass_accuracy(labels_predictions["labels_predictions"], "_")

        print("{} area under ROC: {:.2f}".format(model_identifier, model_areaUnderROC.evaluate(predictions)))
        print("{} area under PR: {:.2f}".format(model_identifier, model_areaUnderPR.evaluate(predictions)))
        print("{} precision: {:.2f}".format(model_identifier, model_precision))
        print("{} recall: {:.2f}".format(model_identifier, model_recall))
        print("{} accuracy: {:.2f}".format(model_identifier, model_accuracy))

        return predictions, trained_model


    def get_decision_tree_feature_importances(feature_names,
                                              dtc_model,
                                              desc_sorting = True):
        '''
        return a list of the form (feature name, feature importance) sorted by the feature importance of the decision tree model
        *feature_names*: list containing the names of the features
        *dtc_model*: trained DecisionTreeClassification model
        *desc_sorting*: boolean flag denoting whether the sorting of the features should be descending regarding the importance of the features
        '''
        dtc_feature_importances = dtc_model.featureImportances.toArray()
        names_feature_importances = list(zip(feature_names, dtc_feature_importances))
        return list(sorted(names_feature_importances, key = lambda k: k[1], reverse = desc_sorting))


    def get_logistic_regression_feature_importances(feature_names,
                                                   lr_model):
        '''
        return one list of the form (feature name, logistic regression's coefficient) sorted by the absolute value of the logistic regression coefficients
        *feature_names*: list containing the names of the features used
        *lr_model*: trained spark.ml logistic regression model
        '''
        coeffs = lr_model.coefficientMatrix.values
        names_coeffs = list(zip(feature_names, coeffs))
        sorted_names_coeffs = list(sorted(names_coeffs, key = lambda k: abs(k[1]), reverse = True))
        return sorted_names_coeffs



    def get_feature_names_from_model_metadata(data,
                                              numeric_predictors_names,
                                              categorical_predictors_names = None):
        '''
        returns a list containing the names of the features used for the training of a spark.ml model in the order that can match their coefficients/importances
        *data*: pyspark dataframe
        *numerical_predictors_names*: list containing the names of the numerical predictors
        *categorical_predictors_names*: list containing the names of the categorical predictors
        '''
        feature_metadata_dict = data.schema[-1].metadata["ml_attr"]["attrs"]
        feature_data_types = feature_metadata_dict.keys()
        feature_names = reduce(lambda names, data_type: names + list(map(lambda feat_metadata: feat_metadata["name"], feature_metadata_dict[data_type])), feature_data_types, [])
        if categorical_predictors_names == None:
            feature_names = list(map(lambda col: "scaled_numerical_predictors_{}".format(numeric_predictors_names[int(col.split("_")[-1])]) if "scaled_numerical_predictors" in col else col, feature_names))
        else:
            feature_names = list(map(lambda col: "scaled_numerical_predictors_{}".format(numeric_predictors_names[int(col.split("")[-1])]) if "scaled_numerical_predictors" in col else "one_hot_{}".format(categorical_predictors_names[int(col.split("")[-1])]) if "one_hot_" in col else col, feature_names))
        return list(map(lambda col: col.strip(), feature_names))



    def cv_acquiring_permutation_performance(data,
                                        model,
                                        categorical_predictors = [],
                                        numerical_predictors = [],
                                        binary_predictors = [],
                                        perm_test_sets_num = 3,
                                        cv_folds = 3,
                                        seed = 2021,
                                        prob_thres = 0.5):
        '''
        returns a list of tuples of the form (feature, permutation feature importance) sorted by the feature importance in descending order
        *data*: spark dataframe containing the dataset
        *model*: spark ml model configuration
        *categorical_predictors*: list containing the categorical predictors
        *numerical_predictors*: list containing the numerical predictors
        *binary_predictors*: list containing the binary predictors
        *perm_test_sets_num*: the number of permuted test sets generated
        *cv_folds*: the number of folds for cross-validation
        *seed*: the seed used during the generation of the cross-validation pairs
        '''

        def generate_shuffled_dataset(test_data, iter_num):
            '''
            returns a shuffled version of the test dataset
            *test_data*: spark dataframe containing the test data
            *iter_num*: the enumeration id of the shuffled test dataframe to be created
            '''
            d = test_data.orderBy(F.rand(seed = iter_num))
            for c in d.columns:
                d = d.withColumnRenamed(c, "sh{}_{}".format(iter_num, c))
            d.createOrReplaceTempView("shuffled_test_data{}".format(iter_num))
            return d


        def generate_join_test_set(train_test):
            '''
            returns a spark dataframe containing the join of the original test set with the different shuffled versions of the original
            *train_test*: TrainTest object
            '''
            test_data = train_test.test_data
            test_data.createOrReplaceTempView("orig_test_data")
            perm_test_sets_enum = list(range(perm_test_sets_num))
            shuffled_datasets = list(map(lambda iter_num: generate_shuffled_dataset(test_data, iter_num), perm_test_sets_enum))

            test_data = spark.sql("select row_number() over (order by 'prty_id') as row_id, * from orig_test_data")

            shuffled_datasets = list(map(lambda iter_num: spark.sql("select row_number() over (order by 'prty_id') as sh{0}_row_id, * from shuffled_test_data{0}".format(iter_num)),
                                                            perm_test_sets_enum))

            shuffled_datasets_enum = list(zip(list(range(len(shuffled_datasets))), shuffled_datasets))
            return reduce(lambda joined_data, shuf_data_enum: joined_data.join(shuf_data_enum[1],
                                                              F.col("row_id") == F.col("sh{}_row_id".format(shuf_data_enum[0]))), shuffled_datasets_enum, test_data)


        def get_fscore_on_predictions(test_predictions):
            '''
            returns the fscore of the predictions on a certain variation of the test set
            *test_predictions*: spark dataframe containing the labels and the predictions of a certain model (dataframe generate with transform function)
            '''
            if prob_thres != 0.5:
                predictions = test_predictions.withColumn("processed_prediction", F.when(sparse_to_array(F.col("probability"))[1] >= prob_thres , 1).otherwise(0))
            else:
                predictions = test_predictions.withColumn("processed_prediction", F.col("prediction"))
            predictions = predictions.withColumn("labels_predictions", F.concat(F.col("label"), F.lit("_"), F.col("processed_prediction").cast(IntegerType())))
            labels_predictions = predictions.select("labels_predictions").toPandas()
            labels_series = pd.Series(labels_predictions["labels_predictions"].str.split("_").str.get(0))
            preds_series = pd.Series(labels_predictions["labels_predictions"].str.split("_").str.get(1))

            f_score = get_fscore_of_label('1', labels_series, preds_series)

            del labels_predictions
            del labels_series
            del preds_series

            return f_score


        def apply_pipeline_and_model_on_shuffled_test_variation(test_data, fitted_pipeline, fitted_model, test_var_iter, feature):
            '''
            returns the fscore for the a particular variation of the test set after permuting one particular feature
            *test_data*: spark dataframe containing the joined original data with its shuffled variations
            *fitted_pipeline*: spark ml pipeline fitted on the training set
            *fitted_model*: spark ml model fitted on the training set
            *test_var_iter*: the identifier of the shuffled variation to be used
            *feature*: the feature whose permutation importance we wish to calculate
            '''
            features_to_select = list(filter(lambda feat: feat != feature, categorical_predictors + numerical_predictors + binary_predictors)) +\
                                                                              ["sh{}_{}".format(test_var_iter, feature)]
            test_data_var = test_data.select(*features_to_select, "label").\
                                      withColumnRenamed("sh{}_{}".format(test_var_iter, feature), feature)
            test_data_var = fitted_pipeline.transform(test_data_var)
            test_predictions = fitted_model.transform(test_data_var)
            var_test_fscore = get_fscore_on_predictions(test_predictions)
            return var_test_fscore


        def apply_pipeline_and_model_on_test_set_and_obtain_error(test_data, fitted_pipeline, fitted_model, feature):
            '''
            returns the average fscore for the different variations of the test set after permuting a particular feature
            *test_data*: spark dataframe containing the joined original data with its shuffled variations
            *fitted_pipeline*: spark ml pipeline fitted on the training set
            *fitted_model*: spark ml model fitted on the training set
            *feature*: the feature whose permutation importance we wish to estimate
            '''
            test_var_iters = list(range(perm_test_sets_num))

            variations_fscores = list(map(lambda var_iter: apply_pipeline_and_model_on_shuffled_test_variation(test_data, fitted_pipeline, fitted_model, var_iter, feature), test_var_iters))
            return sum(variations_fscores) / len(variations_fscores)



        def return_mean_feature_importance_per_feature_for_traintest_pair(train_test, pipeline):
            '''
            returns a tuple of the form (original test set fscore, dictionary of the form (feature name --> mean permutation importance) for a particular train-test pair)
            *train_test*: TrainTest object
            *pipeline*: configured spark ml pipeline
            '''
            all_features = categorical_predictors + numerical_predictors + binary_predictors

            train_data = train_test.train_data
            test_data = train_test.test_data

            fitted_pipeline = pipeline.fit(train_data)

            train_data = fitted_pipeline.transform(train_data)
            train_data.cache()

            trained_model = model.fit(train_data)

            orig_test_data = fitted_pipeline.transform(test_data.select(*all_features, "label"))
            test_predictions = trained_model.transform(orig_test_data)
            orig_test_fscore = get_fscore_on_predictions(test_predictions)
            mean_scores_per_feat = dict(map(lambda feat: (feat, apply_pipeline_and_model_on_test_set_and_obtain_error(test_data, fitted_pipeline, trained_model, feat)), all_features))
            return (orig_test_fscore, mean_scores_per_feat)

        def get_mean_and_std_of_permut_importance_for_feature(train_test_mean_scores, feature):
            '''
            returns a list of tuples of the form (mean of feature's permutation importance, standard deviation of feature's permutation importance)
            *train_test_mean_scores*: list of dictionaries of the form (feature --> mean permutation importance) for a particular training test pair
            *feature*: the feature whose permutation importance we wish to extract
            '''
            feature_importances = np.array(list(map(lambda feat_score_dict: feat_score_dict[0] / feat_score_dict[1][feature], train_test_mean_scores)))
            feat_imp_mean = np.mean(feature_importances)
            feat_imp_std = np.std(feature_importances)
            return (feat_imp_mean, feat_imp_std)


        pipeline = set_up_pipeline(categorical_predictors, numerical_predictors, binary_predictors)

        cross_val_datasets = data.randomSplit(weights = [1.0 for i in range(cv_folds)] , seed = seed)
        train_test_pairs = generate_train_test_pairs(cross_val_datasets)
        train_test_pairs = list(map(lambda train_test: TrainTest(train_test.train_data, generate_join_test_set(train_test)), train_test_pairs))
        train_test_pairs_imp = list(map(lambda train_test: return_mean_feature_importance_per_feature_for_traintest_pair(train_test, pipeline), train_test_pairs))

        feature_fscores = list(map(lambda feat: (feat, get_mean_and_std_of_permut_importance_for_feature(train_test_pairs_imp, feat)), categorical_predictors + numerical_predictors + binary_predictors))

        return sorted(feature_fscores, key=lambda feature_fscore: feature_fscore[1][0], reverse = True)
