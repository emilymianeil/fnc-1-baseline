import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

import utils.generate_test_splits
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, cosine_features
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import tensorflow as tf
from nltk import tokenize
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Embedding, Dense, Dropout, Activation, Lambda
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.dataset import DataSet
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version

# contains hyperparameters for our seq2seq model. you can add some more
hyperparam = {
    'batch_size': 200,
    'max_vocab_size': 20000,
    'max_length': 70,
    'embedding_dim': 100,
    'dropout_rate': 0.3,
    'learning_rate': 0.1,
    'n_epochs': 10,
}

def text_to_padded_seq(data, tokenizer):
    # get word sequences
    word_seq = [text_to_word_sequence(sent) for sent in data]
    # perform text_to_seq and pad
    seq = tokenizer.texts_to_sequences([' '.join(seq[:hyperparam['max_length']]) for seq in word_seq])
    return pad_sequences(seq, maxlen=hyperparam['max_length'], padding='post', truncating='post')


# progress bar
def print_progress(ind, total, message):
    n_bar = 20
    j = round(ind/total,2)
    sys.stdout.write('\r')
    sys.stdout.write(f"   [{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}% {message}")
    sys.stdout.flush()

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_cosine = gen_or_load_feats(hand_features, h, b, "features/cosine."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_cosine]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=5)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout, y_holdout = generate_features(hold_out_stances, d, "holdout")
    for fold in fold_stances:
        Xs[fold], ys[fold] = generate_features(fold_stances[fold], d, str(fold))

    best_score = 0
    best_fold = None

    #Load the training dataset and generate folds
    d = DataSet()

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    # Load Word2Vec
    print("   \nVectorizing data ...")
    articles = d.articles.values()
    sz = len(articles)
    sentences = []
    for article in articles:
        sentences += tokenize.sent_tokenize(article)
    tokens = []
    for idx, sent in enumerate(sentences):
        #print_progress(idx, sz, "")
        words = word_tokenize(sent)
        tokens.append(words)
    print("\n")
    w2v = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)

    best_score = 0
    best_fold = None

    # build tokenizer
    word_seq = [text_to_word_sequence(sent) for sent in sentences]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=hyperparam['max_vocab_size'])
    tokenizer.fit_on_texts([' '.join(seq[:hyperparam['max_length']]) for seq in word_seq])
    word_seq = [text_to_word_sequence(sent) for sent in sentences]

    # build embedding matrix
    vocab_size = len(tokenizer.word_index.items()) + 1
    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, hyperparam['embedding_dim']))
    for i, word in enumerate(tokenizer.word_index.items()):
        word = word[0]
        word = word.lower()
        try:
            embedding_matrix[i] = w2v.wv[word]
        except KeyError as e:
            # skip any words not in w2v model
            continue

    # build model
    e = Embedding(input_dim=len(embedding_matrix),
                  output_dim=hyperparam['embedding_dim'],
                  weights=[embedding_matrix],
                  input_length=hyperparam['max_length'],
                  trainable=False)
    model = Sequential()
    model.add(e)

    model.add(LSTM(128, return_sequences=False))

    model.add(Dense(32))
    model.add(Dropout(rate=0.1))
    model.add(Activation(activation='tanh'))

    model.add(Dense(1, kernel_regularizer='l2'))
    model.add(Dense(4,activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])



    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        X_val = X_test[:(len(X_test) // 2)]
        y_val = y_test[:(len(X_test) // 2)]
        x_test = X_test[(len(X_test) // 2):]
        y_test = y_test[(len(X_test) // 2):]

        # train model
        model.fit(X_train, y_train,
                  batch_size=hyperparam['batch_size'],
                  epochs=1,
                  validation_data=(X_val, y_val),
                  verbose=1)

        predicted = [LABELS[np.argmax(a, axis = 0)] for a in model.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score / max_fold_score

        print("Score for fold " + str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = model

    # Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[np.argmax(a, axis = 0)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual, predicted)
    print("")
    print("")

    # Run on competition dataset
    predicted = [LABELS[np.argmax(a, axis = 0)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted)

