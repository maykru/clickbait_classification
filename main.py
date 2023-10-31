import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from featurize import apply_features
from preprocess import preprocess_split


def parse_args():
    parser = argparse.ArgumentParser(description='clickbait classification')
    parser.add_argument('--alpha', type=float, default=0.1, help="alpha value for Naive Bayes hyperparameter tuning")
    parser.add_argument('--sd', type=str, help="source directory")
    parser.add_argument("--model", type=str, help="Classification model to run. Choose between Naive Bayes ('nb'), "
                                                  "Support Vector Machine ('SVM') and Logistic Regression ('lr')")
    parser.add_argument("--c", type=float, default=1.0, help="c value for Logistic Regression or SVM")
    parser.add_argument("--kernel", type=str, help="")

    args = parser.parse_args()
    return args


def get_text_data(input_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extracts the text data from the input DataFrame.
    :param input_dfs: train, test, val data in DataFrame form
    :return: a Tuple of list, each containing the text data from train, test, val respectively
    """
    train, test, val = input_dfs
    train_text = list(train["headline"])
    test_text = list(test["headline"])
    val_text = list(val["headline"])
    return train_text, test_text, val_text


def get_features(input_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts the feature data from the input DataFrame.
    :param input_dfs: train, test, val data in DataFrame form
    :return: a Tuple of list, each containing the feature data from train, test, val respectively
    """
    train, test, val = input_dfs
    train_features = train.drop(columns=["headline", "clickbait"])
    test_features = test.drop(columns=["headline", "clickbait"])
    val_features = val.drop(columns=["headline", "clickbait"])
    return train_features, test_features, val_features


def vectorize(text_lists: Tuple[List[str], List[str], List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts text data into numerical data using tf-idf.
    :param text_lists: the text data extracted from get_text_data
    :return: tf-idf data stored in DataFrames
    """
    train_text, test_text, val_text = text_lists
    # initialize vectorizer
    vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
    # fit and transform vector matrices
    train_vector_matrix = vectorizer.fit_transform(train_text)
    # train_vector_matrix = vectorizer.transform(train_text)
    test_vector_matrix = vectorizer.transform(test_text)
    val_vector_matrix = vectorizer.transform(val_text)
    # idf data into dataframe
    x_train_idf = pd.DataFrame(data=train_vector_matrix.toarray())
    x_test_idf = pd.DataFrame(data=test_vector_matrix.toarray())
    x_val_idf = pd.DataFrame(data=val_vector_matrix.toarray())
    return x_train_idf, x_test_idf, x_val_idf


def classify(x_train: pd.DataFrame, x_test: pd.DataFrame, labels: List[str], model: str, c=1.0, kernel="linear", alpha=0.1):
    """
    Trains classifier and makes a prediction on the test set.
    :param x_train: train data
    :param x_test: test data
    :param labels: correct labels for train
    :param model: the chosen classification model (Naive Bayes, Logistic Regression or SVM)
    :param c: c value used for Logistic Regression and SVM
    :param kernel: kernel used for SVM
    :param alpha: alpha value used for Naive Bayes
    :return: a list of predicted values between 0 and 1
    """
    if model == "nb":
        classifier = MultinomialNB(alpha=alpha)
    elif model == "lr":
        classifier = LogisticRegression(C=c)
    elif model == "svm":
        classifier = svm.SVC(kernel=kernel, C=c)
    else:
        raise ValueError("Invalid argument. Please specify model as either 'nb', 'lr' or 'svm'.")
    classifier.fit(x_train, labels)
    preds = classifier.predict(x_test)
    return preds


def evaluate(preds_raw: List[float], labels: List[int]):
    """
    Given predictions and gold labels prints out accuracy, precision, recall and f1 score.
    :param preds_raw: labels predicted by the model on the test set
    :param labels: gold labels for the test set
    :return: accuracy, precision, recall, f1 score
    """
    preds = []
    for prediction in preds_raw:
        if prediction < 0.5:
            preds.append(0)
        else:
            preds.append(1)
    print("Accuracy:", metrics.accuracy_score(labels, preds))
    print("Precision:", metrics.precision_score(labels, preds))
    print("Recall:", metrics.recall_score(labels, preds))
    print("F1 Score:", metrics.f1_score(labels, preds))
    return


def main():
    args = parse_args()
    df = apply_features(args.sd)
    train, test, val = preprocess_split(df)
    
    # get features and text data from dataframes
    train_text, test_text, val_text = get_text_data((train, test, val))
    train_features, test_features, val_features = get_features((train, test, val))
    train_idf, test_idf, val_idf = vectorize((train_text, test_text, val_text))

    # combine idf and feaure data into one dataframe for train, test val respectively
    x_train = pd.concat([train_idf, train_features], axis=1)
    x_test = pd.concat([test_idf, test_features], axis=1)
    x_val = pd.concat([val_idf, val_features], axis=1)

    # get labels for classification and evaluation
    train_labels = list(train["clickbait"])
    test_labels = list(test["clickbait"])
    predictions = classify(x_train, x_test, train_labels)
    evaluate(predictions, test_labels)
    return


if __name__ == "__main__":
    main()

