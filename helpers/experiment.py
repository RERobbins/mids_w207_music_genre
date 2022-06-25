import os
import numpy as np
import pandas as pd

from helpers.assess import make_classification_report, make_confusion_matrix
from helpers.split import make_train_test_split, tag_label_feature_split


def experiment(
    model, dataset, name=None, samples_per_genre=None, result_filename=None,
):

    if name is None:
        model_name = type(model).__name__
        if isinstance(dataset, str):
            dataset_name = os.path.basename(dataset).split(".")[0]
        else:
            dataset_name = "anonymous"
        name = f"{model_name}_{dataset_name}"

    print(f"\n\nCommencing Experiment: {name}\n")

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

    model.fit(X_train_std, y_train)

    predictions = model.predict(X_test_std)
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
    )

    print(f"\nTesting accuracy: {test_accuracy = :f}\n")
    make_classification_report(
        y_test,
        y_pred=predictions,
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
        y_pred=predictions,
        label_encoder=le,
        title=f"{dataset_name} test (row normalized)",
    )

    return