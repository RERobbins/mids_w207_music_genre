import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight,compute_class_weight
from tensorflow.keras import backend as K
from helpers.save import results


def make_confusion_matrix(
    y_true,
    y_pred=None,  # optionally pass in precalculated y predictions
    model=None,  # optionally pass in a model (along with x) to autogenerate y_pred
    x=None,  # optionally pass in a feature set (along with model) to autogenerate y_pred
    normalize="true",  # parameter for the sklearn confusion matrix function specifying how to normalize the values
    sample_weight=None,  # optionally provide precalculated sample weights
    autoweight_sample=True,  # generate sample weights automatically before calculating matrix
    print_heatmap=True,  # print the heatmap for confusion matrix
    label_names=None,  # optionally provide an explicit list of label names for the heatmap
    label_encoder=None,  # a label encoder for the to automatically get the label names
    figsize=(9, 9),  # size of the heatmap displayed
    title=None,  # printed title of the heatmap
    ax=None,
    digits=3,
    cbar=True,
    fontsize=10,
    square=True
):

    # resolve polymorphisms / optional values
    y_pred = resolve_y_pred(y_pred=y_pred, model=model, x=x,)
    sample_weight = resolve_sample_weight(y_true, sample_weight, autoweight_sample)
    label_names = resolve_label_names(
        y_true,
        y_pred,
        model=model,
        label_names=label_names,
        label_encoder=label_encoder,
    )

    # calculate the confusion matrix
    cm = confusion_matrix(
        y_true=y_true, y_pred=y_pred, normalize=normalize, sample_weight=sample_weight,
    )

    # print a heatmap
    if print_heatmap:
        if ax is None:
          plt.figure(figsize=figsize, facecolor="white")
        sns.heatmap(
            cm,
            annot=True,
            fmt=f".{digits}f",
            linewidths=0.5,
            square=square,
            cmap="Blues_r",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
            cbar=cbar,
            annot_kws={"fontsize":fontsize}
        )
        if ax:
            ax.set(ylabel="Actual Label")
            ax.set(xlabel="Predicted Label")
        else:
            plt.ylabel("Actual Label")
            plt.xlabel("Predicted Label")
        if title:
            if ax:
                ax.set(title=title)
            else:
                plt.title(title)
        if ax is None:
            plt.show()

    # return the 2D confusion matrix array
    return cm


# reusable helper function to get a y_pred array from various polymorphisms
def resolve_y_pred(y_pred=None, model=None, x=None,postprocess_y_pred_fn=None):
    if y_pred is not None:
        return y_pred
    elif model and x is not None:
        y_pred = model.predict(x)
        if postprocess_y_pred_fn is not None:
            y_pred = postprocess_y_pred_fn(y_pred)
        return y_pred
    else:
        raise Exception(
            "Both a model and x (feature set) must be passed if a y_pred is not provided."
        )


# reusable helper function to get a label_names array from various polymorphisms
def resolve_label_names(
    y_true, y_pred, model=None, label_names=None, label_encoder=None,
):

    # if no label names are explicitly provided
    if not label_names:
        # get a unique list of labels in the matrix
        unique_labels = None
        if model and hasattr(model, 'classes_'):
            unique_labels = list(model.classes_)
        else:
            unique_labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
        if label_encoder:

            # convert label to friendly names if encoder is provided
            label_names = label_encoder.inverse_transform(unique_labels)
        else:
            # otherwise use the raw (usually integer) label values
            label_names = [str(l) for l in unique_labels]
    return label_names


# reusable helper function to get a sample_weight array from various polymorphisms
def resolve_sample_weight(
    y_true, sample_weight=None, autoweight_sample=None,
):
    # compute the weights for each class if not explicitly provided
    if not sample_weight and autoweight_sample:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_true)
    return sample_weight


def make_classification_report(
    # same params as sklearn.metrics.classification_report
    y_true,
    y_pred=None,  # optionally pass in precalculated y predictions
    model=None,  # optionally pass in a model (along with x) to autogenerate y_pred
    x=None,  # optionally pass in a feature set (along with model) to autogenerate y_pred
    labels=None,  # optionally pass a list of label (integers) to *only* include in the report
    target_names=None,  # optionally provide an explicit list of label names defining the label indexes
    label_encoder=None,  # optionally provide a label encoder for the to automatically get the label names
    save_result=False,
    model_name=None,
    dataset_name=None,
    phase=None,
    additional_result_param=None,
    repeat=False,
    sample_weight=None,
    result_filename=None,
    digits=2,
    output_dict=False,
    zero_division="warn",
    print_report=False,
    compare_to=None,
    postprocess_y_pred_fn=None,
):

    if save_result == True:
        if not model_name:
            raise ValueError("Missing model name")
        if not dataset_name:
            raise ValueError("Missing dataset name")
        if not phase:
            raise ValueError("Missing phase")

    # resolve polymorphisms / optional values
    y_pred = resolve_y_pred(y_pred=y_pred, model=model, x=x,postprocess_y_pred_fn=postprocess_y_pred_fn)
    target_names = resolve_label_names(
        y_true,
        y_pred,
        model=model,
        label_names=target_names,
        label_encoder=label_encoder,
    )

    cr = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        sample_weight=sample_weight,
        digits=digits,
        output_dict=True,
        zero_division=zero_division,
    )

    # fix some bizarre parsing irregularity
    if type(cr["accuracy"]) != "dict":
        cr["accuracy"] = {
            "f1-score": cr["accuracy"],
            "support": cr["weighted avg"]["support"],
        }

    # generate mcc for each class
    for i in range(len(target_names)):
        cr[target_names[i]]["mcc"] = matthews_corrcoef(y_true == i, y_pred == i)

    # generate multiclass mcc
    cr["accuracy"]["mcc"] = matthews_corrcoef(y_true, y_pred)

    if save_result:
        save = results(result_filename)
        if additional_result_param:
            save.save(
                model=model_name,
                dataset=dataset_name,
                phase=phase,
                results=cr,
                additional=additional_result_param,
                repeat=repeat,
            )
        else:
            save.save(
                model=model_name,
                dataset=dataset_name,
                phase=phase, 
                results=cr, 
                repeat=repeat
            )

    metric_cols = ["label", "precision", "recall", "f1-score", "support", "mcc"]
    meta_labels = ["", "accuracy", "macro avg", "weighted avg", "min"]
    real_labels = [k for k in cr.keys() if k not in meta_labels]

    cr["min"] = {
        m: min([cr[g][m] for g in real_labels]) for m in metric_cols if m != "label"
    }

    # if another classification report is passed to compare to, calculate the delta for each value
    if compare_to is not None:
        for rk, rv in cr.copy().items():
            for mk, mv in rv.copy().items():
                cr[rk][mk] = mv - compare_to[rk][mk]

    # return dictionary as-is if requested and no string parsing is necessary for print
    # deferred_cr = None
    #if output_dict and print_report == False:
    #    return cr
    #else:
    deferred_cr = cr.copy()

    # parse a table to a string

    # put in empty key to simulate newline been actual labels and metalabels
    cr[""] = {}

    # format all floats as strings
    cr_arr = [
        {
            "label": label,
            **{
                metric_key: (
                    "{:." + str(0 if metric_key == "support" else digits) + "f}"
                ).format(metric_value)
                for metric_key, metric_value in l_object.items()
            },
        }
        for label, l_object in cr.items()
    ]

    # split labels into real labels and meta labels for calculating sort order
    real_labels.sort()

    # sort all table rows
    cr_arr.sort(
        key=lambda x: real_labels.index(x["label"])
        if x["label"] in real_labels
        else len(real_labels) + meta_labels.index(x["label"])
    )

    # prepend header row and simulated newline
    cr_arr = [{c: c for c in metric_cols if c != "label"}, {}] + cr_arr

    # calculate padding necessary to align all rows
    paddings = {
        col: max([len(r.get(col, "")) for r in cr_arr]) + 2 for col in metric_cols
    }

    report_string = "\n".join(
        [
            "".join([r.get(c, "").rjust(paddings[c], " ") for c in metric_cols])
            for r in cr_arr
        ]
    )

    if print_report:
        print(report_string)
    
    deferred_cr['text_summary'] = report_string

    return deferred_cr

def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
    return deferred_cr if output_dict else report_string

def make_class_weight_dict(y,le):
    return {
        i: c
        for i, c in enumerate(
            compute_class_weight(
                class_weight="balanced", classes=le.transform(le.classes_), y=y
            )
        )
    }

def make_notebook_classification_summary(y_true,y_pred,le=None,title=None):
    
    cr = make_classification_report(y_true, y_pred, label_encoder=le, print_report=False,output_dict=True)

    f, axs = plt.subplots(1, 2, figsize=(14, 6))
    f.set(facecolor='white')
    axs[0].set(facecolor='white')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].text(
        0.90, 0.90, cr['text_summary'], family='monospace',
        horizontalalignment='right',
        verticalalignment='top',
        # transform=axs[1].transAxes
    )
    if title is not None:
        plt.suptitle(title,color='black',size=18)

    make_confusion_matrix(y_true, y_pred, label_encoder=le, ax=axs[1],digits=2)

    plt.show()
    
def make_notebook_undersampling_smote_classification_summary(
    y_true_test,
    undersample_classifier_y_pred_test,
    smote_classifier_y_pred_test,
    undersample_train_results,
    undersample_validation_results,
    smote_train_results,
    smote_validation_results,
    le=None,title=None
):
    
    undersample_cr = make_classification_report(
        y_true_test,
        undersample_classifier_y_pred_test,
        label_encoder=le,
        print_report=False,
        output_dict=True
    )['text_summary']
    
    smote_cr = make_classification_report(
        y_true_test,
        smote_classifier_y_pred_test,
        label_encoder=le,
        print_report=False,
        output_dict=True
    )['text_summary']

    f, axs = plt.subplots(4, 2, figsize=(14, 22))
    f.set(facecolor='white')
    
    
    axs[0][0].plot(undersample_train_results['mlogloss'], label='train')
    axs[0][0].plot(undersample_validation_results['mlogloss'], label='validate')
    axs[0][0].set(title='Undersample Training')
    axs[0][0].set(xlabel='epoch')
    axs[0][0].set(label='mlogloss')
    axs[0][0].legend()
    
    axs[0][0].set_ylim([0, 2.25])
    axs[0][1].set_ylim([0, 2.25])

    axs[1][0].plot(undersample_train_results['auc'], label='train')
    axs[1][0].plot(undersample_validation_results['auc'], label='validate')
    axs[1][0].set(title='Undersample Evaluation')
    axs[1][0].set(xlabel='epoch')
    axs[1][0].set(label='ROC AUC')
    axs[1][0].legend()

    axs[0][1].plot(smote_train_results['mlogloss'], label='train')
    axs[0][1].plot(smote_validation_results['mlogloss'], label='validate')
    axs[0][1].set(title='SMOTE Training')
    axs[0][1].set(xlabel='epoch')
    axs[0][1].set(label='mlogloss')
    axs[0][1].legend()

    axs[1][1].plot(smote_train_results['auc'], label='train')
    axs[1][1].plot(smote_validation_results['auc'], label='validate')
    axs[1][1].set(title='SMOTE Evaluation')
    axs[1][1].set(xlabel='epoch')
    axs[1][1].set(label='ROC AUC')
    axs[1][1].legend()
    
    axs[1][0].set_ylim([0.8, 1.005])
    axs[1][1].set_ylim([0.8, 1.005])
    
    for i in range(2):
        axs[2,i].set(facecolor='white')
        axs[2,i].set_yticklabels([])
        axs[2,i].set_xticklabels([])
        axs[2,i].text(
            0.90, 0.90,
            undersample_cr if i == 0 else smote_cr,
            family='monospace',
            horizontalalignment='right',
            verticalalignment='top',
            # transform=axs[1].transAxes
        )
        axs[2,i].set_title(('Undersample' if i == 0 else 'SMOTE')+' Classification Report',pad=-14)
    
    make_confusion_matrix(y_true_test, undersample_classifier_y_pred_test, label_encoder=le,
                          ax=axs[3][0],
                          digits=2,cbar=True,fontsize=8,title="Undersample Confusion Matrix",square=True)
    make_confusion_matrix(y_true_test, smote_classifier_y_pred_test, label_encoder=le,
                          ax=axs[3][1],
                          digits=2,cbar=True,fontsize=8,title="SMOTE Confusion Matrix",square=True)
    
    if title is not None:
        plt.suptitle(title,color='black',size=18)

    #make_confusion_matrix(y_true, y_pred, label_encoder=le, ax=axs[1],digits=2)

    plt.show()










