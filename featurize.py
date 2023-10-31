import re
import pandas as pd


def is_num(headline: str) -> int:
    """
    check if given headline starts with 1- or 2-digit number
    :param headline: the headline to be examined
    :return: True if starts with 1/2-digit number, False if not
    """
    if re.match(r"^[0-9]{1,2}[ ]", headline):
        return 1
    return 0


def is_question(headline: str) -> int:
    """
    check if headline is question, i.e. starts with question word (or other clickbaity words)
    :param headline: headline to be examined
    :return: True (1) if is question, False (0) if not
    """
    question_words = ["Who", "What", "What's", "When", "Where", "Why", "Which",
                      "How", "Do", "Should", "Are", "Can", "If", "Here", "Here's"]
    headline_list = headline.split(" ")
    if headline_list[0] in question_words:
        return 1
    return 0


def apply_features(input_dir: str) -> pd.DataFrame:
    """
    apply the methods above to the input data
    :param input_dir: path to directory where original data is stored
    :return: text data + features organized in a DataFrame
    """
    path = input_dir + r"\clickbait_data.csv"
    df = pd.read_csv(path, encoding="utf-8", sep=",")
    df["is_num"] = df["headline"].apply(is_num)
    df["is_q"] = df["headline"].apply(is_question)
    return df
