# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MNLI_LABEL = ['entailment', 'neutral', 'contradiction',
              'entailment\'', 'neutral\'', 'contradiction\'']
EQ_LABEL = ['equivalent', 'not_equivalent', 'equivalent\'', 'not_equivalent\'']
ENTAIL_LABEL = ['entailment', 'not_entailment', 'entailment\'',
                'not_entailment\'', '0', '1', '0\'', '1\'']

LABEL_SET = {
    # 'positive\'', 'negative\'' is used for label constraint due to a bug of TextAttack repo.
    'sst2': ['positive', 'negative', 'positive\'', 'negative\'', '0', '1', '0\'', '1\''],
    'mnli': MNLI_LABEL,
    'mnli_mismatched': MNLI_LABEL,
    'mnli_matched': MNLI_LABEL,
    'qqp': EQ_LABEL,
    'qnli': ENTAIL_LABEL,
    'rte': ENTAIL_LABEL,
    'cola': ['unacceptable', 'acceptable', 'unacceptable\'', 'acceptable\''],
    'mrpc': EQ_LABEL,
    'wnli': ENTAIL_LABEL,
    'mmlu': ['A', 'B', 'C', 'D', 'A\'', 'B\'', 'C\'', 'D\'', 'a', 'b', 'c', 'd', 'a\'', 'b\'', 'c\'', 'd\''],
    'crass': ['A', 'B', 'C', 'A\'', 'B\'', 'C\'', 'a', 'b', 'c', 'a\'', 'b\'', 'c\''],
}

GENERATE_LEN = 7

LABEL_TO_ID = {
    'mmlu': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'},
    'crass': {'A': 'A', 'B': 'B', 'C': 'C', 'a': 'a', 'b': 'b', 'c': 'c'},
    'sst2': {'negative': 0, 'positive': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'mnli_mismatched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'mnli_matched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'qqp': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1},
    'qnli': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'rte': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'cola': {'unacceptable': 0, 'acceptable': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mrpc': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1},
    'wnli': {'entailment': 1, 'not_entailment': 0, '0': 0, '1': 1, 0: 0, 1: 1},
}

ID_TO_LABEL = {
    'mmlu': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'},
    'crass': {'A': 'A', 'B': 'B', 'C': 'C', 'a': 'a', 'b': 'b', 'c': 'c'},
    'sst2': {0: 'negative', 1: 'positive'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_matched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_mismatched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'qqp': {1: 'equivalent', 0: 'not_equivalent'},
    'qnli': {0: 'entailment', 1: 'not_entailment'},
    'rte': {0: 'entailment', 1: 'not_entailment'},
    'cola': {0: 'unacceptable', 1: 'acceptable'},
    'mrpc': {1: 'equivalent', 0: 'not_equivalent'},
    'wnli': {1: 'entailment', 0: 'not_entailment'},
}


OOD_clean_instructions = {
    'sst2': [
        "Read the provided excerpt and choose between 'positive' and 'negative' to describe its sentiment: ",
        "Analyze the tone of this statement and respond with either 'positive' or 'negative': ",
        "Evaluate the sentiment of the given text and classify it as 'positive' or 'negative': ",
        "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: ",
        "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to classify this statement: ",
        "Acting as a sentiment evaluator, identify if the given sentence is 'positive' or 'negative'. Classify: ",
    ],
    'qqp': [
        "Can these two statements be considered equal in meaning? Answer with 'equivalent' or 'not_equivalent': ",
        'Are the following two questions equivalent or not? Answer me with "equivalent" or "not_equivalent". ',
        "Determine if the given pair of statements can be considered the same by responding with 'equivalent' or 'not_equivalent'. ",
        "In your role as a question comparison tool, assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent'. ",
        "As a question equivalence detection system, examine the provided questions and respond with 'equivalent' if they are the same in meaning, or 'not_equivalent' if they are different. ",
        "Functioning as a question similarity evaluation tool, analyze the given questions and decide if they share the same meaning, responding with 'equivalent' or 'not_equivalent'. ",
    ],
    'mnli': [
        "Assess the connection between the following sentences and classify it as 'entailment', 'neutral', or 'contradiction': ",
        "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':",
        "Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. Answer with either 'entailment', 'neutral', or 'contradiction':",
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment', 'neutral', or 'contradiction':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment', 'neutral', or 'contradiction':",
    ],
    'qnli': [
        "Consider the context and question, and indicate if the answer can be logically deduced from the context by responding with 'entailment' or 'not_entailment'.",
        "Given the question and context provided, determine if the answer can be inferred by choosing 'entailment' or 'not_entailment'. ",
        "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'. ",
        "As a language expert, assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment'. ",
        "In your role as a semantic evaluator, determine if the provided context justifies the answer to the question and answer with 'entailment' or 'not_entailment'. ",
        "As a textual analyst, examine if the given context logically implies the answer to the question and indicate your decision with 'entailment' or 'not_entailment'. ",
    ],
    'rte': [
        "Determine if the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment'. ",
        'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment", just one word. ',
        "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment'.",
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment':",
    ],
    'cola': [
        "Review the sentence below and identify whether its grammar is 'Acceptable' or 'Unacceptable': ",
        "Assess the following sentence and determine if it is grammatically correct. Respond with 'Acceptable' or 'Unacceptable':",
        "Examine the given sentence and decide if it is grammatically sound. Answer with either 'Acceptable' or 'Unacceptable':",
        "In your role as a grammar check tool, assess the following sentence and classify it as 'acceptable' if it is grammatically correct or 'unacceptable' if it is incorrect:",
        "As a grammar identification system, examine the provided sentence and respond with 'acceptable' for grammatically correct sentences or 'unacceptable' for incorrect ones:",
        "Functioning as a grammar evaluation tool, analyze the given sentence and decide if it is grammatically correct, responding with 'acceptable' or 'unacceptable':",
    ],
    'mrpc': [
        "Can the given sentences be considered semantically identical? Please reply with 'equivalent' or 'not_equivalent'. ",
        "Do these two sentences have the same underlying meaning? Respond with 'equivalent' or 'not_equivalent'. ",
        "Are the meanings of the following pair of sentences the same? Answer with 'equivalent' or 'not_equivalent'. ",
        "As a semantic comparison expert, evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent'. ",
        "In your capacity as a language analyst, assess the following sentences and classify their similarity as 'equivalent' or 'not_equivalent'. ",
        "As a sentence similarity evaluator, analyze the provided sentences and indicate if their meanings are 'equivalent' or 'not_equivalent'. ",
    ],
    'wnli': [
        "Identify whether the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment'. ",
        'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment", just one word. ',
        "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment'.",
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment':",
    ],
    'mmlu': [
        "Answer the following multiple-choice question about {} by selecting the correct option: 'A', 'B', 'C', or 'D'. ",
        "For the multiple-choice question related to {}, please choose the most accurate answer from 'A', 'B', 'C', or 'D'. ",
        "Below are multiple-choice question concerning {}. Indicate your response with 'A', 'B', 'C', or 'D'. ",
        "As an expert in {}, respond to the following multiple-choice question by selecting 'A', 'B', 'C', or 'D'.",
        "Given your proficiency in {}, please answer the subsequent multiple-choice question with 'A', 'B', 'C', or 'D'.",
        "With your knowledge of {}, tackle the following multiple-choice question by choosing 'A', 'B', 'C', or 'D'.",
    ],
    'crass': [
        "Based on common sense reasoning, read each hypothetical scenario and select the most appropriate answer from 'A', 'B', or 'C'. ",
        "Utilize common sense logic to analyze each given hypothetical situation and choose the most fitting response from options 'A', 'B', or 'C'.",
        "Employ common reasoning to evaluate each presented imaginary scenario and determine the best choice among 'A', 'B', or 'C'.",
        "As a common sense reasoning analyst, review each hypothetical scenario and determine the most suitable choice among 'A', 'B', or 'C'. ",
        "As an evaluator of common sense logic, examine each presented hypothetical case and identify the most fitting option from 'A', 'B', or 'C'.",
        "As a professional in common sense reasoning, scrutinize each proposed hypothetical and identify the most relevant selection from 'A', 'B', or 'C'."
    ],
}