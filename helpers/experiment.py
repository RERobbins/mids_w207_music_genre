import os
import numpy as np
import pandas as pd

from helpers.assess import make_classification_report, make_confusion_matrix, resolve_sample_weight
from helpers.split import make_train_test_split, tag_label_feature_split
from sklearn.utils.class_weight import compute_class_weight

def experiment(
    model,
    dataset,
    name=None,
    samples_per_genre=None,
    result_filename=None,
    model_fit_call_fn=None, # a custom fn to fall model.fit if the model type does not accept the default parameters
    postprocess_y_pred_fn=None # if the model type requires a certain transformation of y_pred before scoring, pass the transformation fn here
):

    name,dataset_name = resolve_experiment_names(name,model,dataset)

    if callable(model) == False:
        print(f"\n\nCommencing Experiment: {name}\n")
    # if the model is a factory, we cannot necessarily name the model yet
    elif name is not None:
        print(f"\n\nPreparing model for Experiment: {name}\n")
    else:
        print(f"\n\nPreparing model for Experiment on dataset: {dataset_name}\n")

    if isinstance(dataset, str):
        df = pd.read_pickle(dataset)
    else:
        df = dataset

    _, (y, le), X = tag_label_feature_split(
        df, label_format="encoded", samples_per_genre=samples_per_genre
    )

    X_train_std, X_test_std, y_train, y_test = make_train_test_split(
        X, y, test_size=0.2, random_state=10, stratify=y, x_scaler="standard"
    )

    # if the model is actually a model factory function, then invoke to build model
    if callable(model):
        model = model(X_train=X_train_std,y_train=y_train,le=le)
        # can now generate the name of the model
        name,_ = resolve_experiment_names(name,model,dataset)
        print(f"\n\nCommencing Experiment: {name}\n")

    # if the model requires a custom invokation to the fit method, call it here
    if model_fit_call_fn is not None:
        model_fit_call_fn(
            model=model,
            X_train=X_train_std,
            y_train=y_train,
            class_weight={i:c  for i,c in enumerate(compute_class_weight(class_weight='balanced',classes=le.transform(le.classes_),y=y_train))}
        )
    # otherwise use the default fit method
    else:
        model.fit(X_train_std, y_train)

    y_test_pred = model.predict(X_test_std)
    if postprocess_y_pred_fn is not None:
        y_test_pred = postprocess_y_pred_fn(y_test_pred)
    train_accuracy = float("nan")
    test_accuracy = float("nan")
    if hasattr(model,'score'):
        train_accuracy = model.score(X_train_std, y_train)
        test_accuracy = model.score(X_test_std, y_test)

    print(f"Training accuracy: {train_accuracy = :f}\n")
    make_classification_report(
        y_train,
        model=model,
        x=X_train_std,
        digits=4,
        label_encoder=le,
        print_report=True,
        save_result=result_filename is not None,
        result_filename=result_filename,
        model_name=name + "_train",
        repeat=True,
        postprocess_y_pred_fn=postprocess_y_pred_fn,
    )

    print(f"\nTesting accuracy: {test_accuracy = :f}\n")
    make_classification_report(
        y_test,
        y_pred=y_test_pred,
        digits=4,
        label_encoder=le,
        print_report=True,
        save_result=result_filename is not None,
        result_filename=result_filename,
        model_name=name + "_test",
        repeat=True,
    )

    make_confusion_matrix(
        y_test,
        y_pred=y_test_pred,
        label_encoder=le,
        title=f"{dataset_name} test (row normalized)",
    )

    return

def resolve_experiment_names(name,model,dataset_name):
    # get dataset name
    if isinstance(dataset_name, str):
        dataset_name = os.path.basename(dataset_name).split(".")[0]
    else:
        dataset_name = "anonymous"
    # parse model name if not explicitly defined
    if name is None and model is not None:
        model_name = type(model).__name__
        name = f"{model_name}_{dataset_name}"
    return name,dataset_name