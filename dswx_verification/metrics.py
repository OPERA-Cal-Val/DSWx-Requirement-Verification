import numpy as np
import pandas as pd
import sklearn.metrics

from .constants import CLASS_DICT as class_dict


def get_acc_per_class_for_one_trial(y_val, y_dswx):
    acc_per_class = {}
    for c in [0, 1, 2]:
        y_val_temp = y_val.copy()
        y_dswx_temp = y_dswx.copy()

        y_val_temp[y_val_temp != c] = 255
        y_dswx_temp[y_dswx_temp != c] = 255

        acc_per_class[f'acc_per_class.{class_dict[c]}'] = (y_val_temp == y_dswx_temp).sum() / y_dswx.size
    return acc_per_class


def get_binary_water_acc_for_one_trial(y_val, y_dswx):

    y_val_temp = y_val.copy()
    y_dswx_temp = y_dswx.copy()

    y_val_temp[~np.isin(y_val_temp, [1, 2])] = 255
    y_val_temp[np.isin(y_val_temp, [1, 2])] = 1

    y_dswx_temp[~np.isin(y_dswx_temp, [1, 2])] = 255
    y_dswx_temp[np.isin(y_dswx_temp, [1, 2])] = 1

    binary_water_acc = (y_val_temp == y_dswx_temp).sum() / y_dswx.size
    return binary_water_acc


def get_prec_recall_score_for_one_trial(y_val, y_dswx):

    prec, recall, f1, supp = sklearn.metrics.precision_recall_fscore_support(y_val,
                                                                             y_dswx,
                                                                             labels=[0, 1, 2],
                                                                             # if there are no classes
                                                                             # Assume "perfect"
                                                                             zero_division=1
                                                                             )

    recall_per_class = {class_dict[label]: recall[label] for label in [0, 1, 2]}
    prec_per_class = {class_dict[label]: prec[label] for label in [0, 1, 2]}
    f1_per_class = {class_dict[label]: f1[label] for label in [0, 1, 2]}
    supp_per_class = {class_dict[label]: int(supp[label]) for label in [0, 1, 2]}
    binary_water_acc = get_binary_water_acc_for_one_trial(y_val, y_dswx)
    return {
            'precision': prec_per_class,
            'recall': recall_per_class,
            'f1_per_class': f1_per_class,
            'supp_per_class': supp_per_class,
            'binary_water_acc': binary_water_acc}


def get_confusion_matrix_for_one_trial(y_val, y_dswx):
    y_dswx_str = pd.Series([class_dict[class_id] for class_id in y_dswx], name='OPERA_DSWx')
    y_val_str = pd.Series([class_dict[class_id] for class_id in y_val], name='OPERA_Validation')
    df_conf = pd.crosstab(y_val_str, y_dswx_str)
    df_conf_formatted = df_conf.astype(int)

    name = df_conf.index.name
    df_conf_formatted.rename(index={index: f'{index}_{name}' for index in df_conf.index}, inplace=True)
    col_name = df_conf.columns.name
    df_conf_formatted.rename(columns={col: f'{col}_{col_name}' for col in df_conf.columns}, inplace=True)
    return df_conf_formatted


def get_all_metrics_for_one_trial(y_val, y_dswx):
    total_acc = sklearn.metrics.accuracy_score(y_val, y_dswx)

    pr_dict = get_prec_recall_score_for_one_trial(y_val, y_dswx)
    acc_per_class = get_acc_per_class_for_one_trial(y_val, y_dswx)
    df_conf_formatted = get_confusion_matrix_for_one_trial(y_val, y_dswx)

    return {'total_accuracy': total_acc,
            'confusion_matrix': df_conf_formatted.to_dict(),
            **pr_dict,
            **acc_per_class}
