from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd

def calculate_bleu_scores(row):
    """
    Calculate BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4) for a given row of a DataFrame.

    Args:
        row (pd.Series): A row from the DataFrame with the following structure:
                         - 'caption': A list of reference captions (list of strings).
                         - 'predict': A single predicted caption (string).

    Returns:
        pd.Series: A series containing BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.

    Example:
        # Assuming df_test is a DataFrame with 'caption' and 'predict' columns
        df_test[['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']] = df_test.apply(calculate_bleu_scores, axis=1)
    """
    reference = [caption.split() for caption in row['caption']]
    prediction = row['predict'].split()

    smoothie = SmoothingFunction().method4

    bleu_1 = sentence_bleu(reference, prediction, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_3 = sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return pd.Series([bleu_1, bleu_2, bleu_3, bleu_4])

def calculate_rouge_scores(row):
    """
    Calculate the average ROUGE-L F1 score for a given row of a DataFrame.

    Args:
        row (pd.Series): A row from the DataFrame with the following structure:
                         - 'caption': A list of reference captions (list of strings).
                         - 'predict': A single predicted caption (string).

    Returns:
        float: The average ROUGE-L F1 score across all reference captions.

    Example:
        # Assuming df_test is a DataFrame with 'caption' and 'predict' columns
        df_test['rouge_L'] = df_test.apply(calculate_rouge_scores, axis=1)
    """
    reference = row['caption']
    prediction = row['predict']

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_f1 = 0

    for ref in reference:
        scores = scorer.score(ref, prediction)
        rouge_l = scores['rougeL']
        total_f1 += rouge_l.fmeasure

    avg_f1 = total_f1 / len(reference)

    return avg_f1
