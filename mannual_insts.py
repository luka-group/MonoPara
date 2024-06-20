# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

LABEL_SET = {
    # 'positive\'', 'negative\'' is used for label constraint due to a bug of TextAttack repo.
    'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'],
    'imdb': ['negative', 'positive'],
    'gimmaru/newspop': ['microsoft', 'obama', 'economy', 'palestine'],
    'cola': ["Inappropriate", "Appropriate"],
    'dair-ai/emotion': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
    'offensive': ["no", "yes"],
    'sst2': ["negative", "positive"],
    'fancyzhx/dbpedia_14': ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"],
    'mnli': [],
    'mrpc': [],
}


mannual_instructions = {
    'ag_news': [
        "What label best describes this news article?",
        "What is this piece of news regarding?",
        "Which newspaper section would this article likely appear in?",
        "What topic is this news article about?",
    ],
    'imdb': [
        "This movie review expresses what sentiment?",
        "Did the reviewer find this movie good or bad?",
        "Is this review positive or negative?",
        "How does the viewer feel about the movie?",
        "What sentiment does the writer express for the movie?",
        "What sentiment is expressed for the movie?",
        "What is the sentiment expressed in this text?",
        "Did the reviewer enjoy the movie?",
        "What is the sentiment expressed by the reviewer for the movie?",
        "How does the reviewer feel about the movie?",
    ],
}