from . import *

class Evaluation:

    '''
    contains functions that are responsible for evaluate the predictions of a certain supervised  machine learning model
    '''

    def get_precision_of_label(label, labels, predictions):
        '''
        returns the precision associated with the predictions of a particular label
        *label*: the identifier of the label
        *labels*: pandas series containing the labels
        *predictions*: pandas series containing the predictions
        '''
        label_pred_index = predictions[predictions == label].index
        corresponding_labels_to_label_predictions = labels.loc[label_pred_index]
        correct_count = len(corresponding_labels_to_label_predictions[corresponding_labels_to_label_predictions == label])
        precision = correct_count / len(label_pred_index) if len(label_pred_index) > 0 else 0
        return precision


    def get_recall_of_label(label,
                            labels,
                            predictions):
        '''
        returns the recall associated with the predictions of a particular label
        label: the identifier of the label
        labels: pandas series containing the labels
        predictions: pandas series containing the predictions
        '''
        label_pred_index = predictions[predictions == label].index
        corresponding_labels_to_label_predictions = labels.loc[label_pred_index]
        correct_count = len(corresponding_labels_to_label_predictions[corresponding_labels_to_label_predictions == label])
        labels_count = len(labels[labels == label])
        recall = correct_count /labels_count if labels_count > 0 else 0
        return recall


    def get_fscore_of_label(label,
                            labels,
                            predictions):
        '''
        returns the f-score associated with the predictions of a particular label
        label: the identifier of the label
        labels: pandas series containing the labels
        predictions: pandas series containing the predictions
        '''
        label_pred_index = predictions[predictions == label].index
        corresponding_labels_to_label_predictions = labels.loc[label_pred_index]
        labels.index = list(labels.index)
        non_corresponding_labels_to_label_predictions = labels.loc[~labels.index.isin(label_pred_index)]
        true_positive_count = len(corresponding_labels_to_label_predictions[corresponding_labels_to_label_predictions == label])
        false_positive_count = len(corresponding_labels_to_label_predictions) - true_positive_count
        false_negative_count = len(non_corresponding_labels_to_label_predictions[non_corresponding_labels_to_label_predictions == label])
        f_score = true_positive_count / (true_positive_count + 0.5 * (false_positive_count + false_negative_count))
        return f_score


    def get_multiclass_accuracy(labels_predictions,
                                delim):
        '''
        returns the multi-class classification's accuracy
        labels_predictions: the pandas series containing the labels and the predictions separated by a certain delimiter
        delim: the character used as delimiter between the labels and the predictions
        '''
        correct_count = (labels_predictions.str.split("").str[0] == labels_predictions.str.split("").str[1]).sum()
        return correct_count / len(labels_predictions)
