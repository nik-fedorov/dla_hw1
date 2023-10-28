import editdistance


def calc_cer(target_text, predicted_text) -> float:
    denominator = len(target_text) or 1
    return editdistance.eval(target_text, predicted_text) / denominator


def calc_wer(target_text, predicted_text) -> float:
    target_text_splitted = target_text.split()
    predicted_text_splitted = predicted_text.split()
    denominator = len(target_text_splitted) or 1
    return editdistance.eval(target_text_splitted, predicted_text_splitted) / denominator
