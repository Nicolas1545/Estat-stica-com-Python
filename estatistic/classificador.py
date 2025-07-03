import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

sms['message_clean'] = sms['message'].str.replace(r'[^\w\s]+', '', regex=True).str.lower()

vocabulary = set(''.join(sms['message_clean']).split())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sms['message_clean'], sms['label_num'], random_state=1)

p_spam = y_train.mean()
p_ham = 1 - p_spam

train_spam_messages = X_train[y_train == 1]
train_ham_messages = X_train[y_train == 0]

n_words_spam = len(''.join(train_spam_messages).split())
n_words_ham = len(''.join(train_ham_messages).split())

spam_word_probs = {word: 0 for word in vocabulary}
ham_word_probs = {word: 0 for word in vocabulary}

alpha = 1

for word in vocabulary:
    count = ''.join(train_spam_messages).count(word)
    spam_word_probs[word] = (count + alpha) / (n_words_spam + alpha * len(vocabulary))

for word in vocabulary:
    count = ''.join(train_ham_messages).count(word)
    ham_word_probs[word] = (count + alpha) / (n_words_ham + alpha * len(vocabulary))

import numpy as np

def classify(message):
    words = message.split()

    log_prob_spam = np.log(p_spam)
    for word in words:
        if word in vocabulary:
            log_prob_spam += np.log(spam_word_probs[word])
    
    log_prob_ham = np.log(p_ham)
    for word in words:
        if word in vocabulary:
            log_prob_ham += np.log(ham_word_probs[word])
    
    if log_prob_spam > log_prob_ham:
        return 1
    else:
        return 0
    
predictions = X_test.apply(classify)
accuracy = (predictions == y_test).mean()
print(f"Äcurácia do nosso classificador Naive Bayes: {accuracy:.2%}")