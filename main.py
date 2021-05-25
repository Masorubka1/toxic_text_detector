import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import pickle
import os
import time


def get_vectorizer(all_text, train_text, test_text):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000,
        dtype=np.float32)
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000,
        dtype=np.float32)
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)
    
    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])
    
    with open('word_vectorizer.pk', 'wb') as fin:
        pickle.dump(word_vectorizer, fin)
    with open('char_vectorizer.pk', 'wb') as fin:
        pickle.dump(char_vectorizer, fin)

    return (train_features, test_features)


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#train = pd.read_csv('train.csv').fillna(' ')
#test = pd.read_csv('test.csv').fillna(' ')
#train.to_csv('modified_train.csv', index=False)
#test.to_csv('modified_test.csv', index=False)

train = pd.read_csv('modified_train.csv')
test = pd.read_csv('modified_test.csv')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

if not os.path.exists('word_vectorizer.pk') or not os.path.exists('char_vectorizer.pk'):
    train_features, test_features = get_vectorizer(all_text, train_text, test_text)
else:
    with open('word_vectorizer.pk', 'rb') as fi:
        word_vectorizer = pickle.load(fi)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    
    with open('char_vectorizer.pk', 'rb') as fi:
        char_vectorizer = pickle.load(fi)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)
    
    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])
    
scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)
