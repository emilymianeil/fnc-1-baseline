import sys
import numpy as np
import math

from sklearn.ensemble import GradientBoostingClassifier

import utils.generate_test_splits
import xgboost as xgb
from xgboost import XGBClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import tensorflow as tf
from keras.layers import Layer
import keras.backend as K
from nltk import tokenize
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Embedding, GlobalMaxPooling1D, Dense, Conv1D, SimpleRNN, Flatten, Dropout, Activation, Lambda, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPool1D, SpatialDropout1D
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import regularizers
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
    'max_length': 44,
    'embedding_dim': 100,
    'dropout_rate': 0.3,
    'learning_rate': 0.1,
    'n_epochs': 10,
}

def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

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

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
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
    values = d.stances
    sz = len(articles)
    sentences = []
    for article in articles:
        sentences += tokenize.sent_tokenize(article)
    for val in values:
        sentences += tokenize.sent_tokenize(val['Headline'])
    tokens = []
    for idx, sent in enumerate(sentences):
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

    tkn = []
    [tkn.append(x) for x in tokens if x not in tkn]
    #vocab_size = len(tokenizer.word_index.items()) + 1
    vocab_size = len(tkn) + 1
    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, hyperparam['embedding_dim']))
    total = 0
    missed = 0
    for i, word in enumerate(tkn):
        word = word[0]
        word = word.lower()
        total += 1
        try:
            embedding_matrix[i] = w2v.wv[word]
        except KeyError as e:
            missed += 1
            # skip any words not in w2v model
            continue

    # build model
    e = Embedding(input_dim=len(embedding_matrix),
                  output_dim=hyperparam['embedding_dim'],
                  weights=[embedding_matrix],
                  input_length=hyperparam['max_length'],
                  trainable=False)

    modelRel = Sequential()
    modelRel.add(e)
    modelRel.add(Bidirectional(LSTM(128, dropout=0.3)))
    modelRel.add(Activation(activation='relu'))
    modelRel.add(Dense(1, kernel_regularizer='l2'))
    modelRel.add(Dense(1, activation="sigmoid"))
    modelRel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    modelType = Sequential()
    modelType.add(e)
    modelType.add(SimpleRNN(15, return_sequences=True))
    modelType.add(SimpleRNN(15))
    # modelType.add(Bidirectional(LSTM(128, dropout=0.6)))
    # modelType.add(Activation(activation='relu'))
    # modelType.add(Dense(1, kernel_regularizer='l2'))
    modelType.add(Dense(3, activation='softmax'))
    modelType.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    # Classifier for each fold
    t = 0
    for fold in fold_stances:
        '''if t >0:
            break'''
        t = 1
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
        X_test = X_test[(len(X_test) // 2):]
        y_test = y_test[(len(y_test) // 2):]

        y_train_rel = []
        y_train_type = []
        X_train_type = []
        for i, y in enumerate(y_train):
            if y < 3:
                y_train_rel.append(0)
                X_train_type.append(X_train[i])
                y_train_type.append(y)
            else:
                y_train_rel.append(1)
        y_val_rel = []
        X_val_type = []
        y_val_type = []
        for i, y in enumerate(y_val):
            if y < 3:
                y_val_rel.append(0)
                X_val_type.append(X_val[i])
                y_val_type.append(y)
            else:
                y_val_rel.append(1)

        y_test_rel = []
        X_test_type = []
        y_test_type = []

        for i, y in enumerate(y_test):
            if y < 3:
                y_test_rel.append(0)
                X_test_type.append(X_test[i])
                y_test_type.append(y)
            else:
                y_test_rel.append(1)

        # train model
        print("START")# X_shortened_train = np.asarray(X_shortened_train).astype('float32').reshape((-1, 1))
        y_val_rel = np.asarray(y_val_rel).astype('float32').reshape((-1, 1))
        modelRel.fit(X_train, np.asarray(y_train_rel),  # class_weight=class_weights,
                     batch_size=hyperparam['batch_size'],
                     epochs=1,
                     validation_data=(X_val, np.asarray(y_val_rel)),
                     verbose=1)

        # train mode
        class_weights = {0: 1.0, 1: 5, 2: 1.0}

        modelType.fit(np.asarray(X_train_type), np.asarray(y_train_type),  class_weight=class_weights,
                     batch_size=hyperparam['batch_size'],
                     epochs=1,
                     validation_data=(np.asarray(X_val_type), np.asarray(y_val_type)),
                     verbose=1)

        print("DONE")


    # Run on Holdout set and report the final score on the holdout set
    #predicted = [LABELS[np.argmax(a, axis = 0)] for a in modelRel.predict(X_holdout)]
    predsRel = modelRel.predict(X_holdout)
    X_holdout_rel = []
    numRelated = 0
    numUnrelated = 0
    for i, p in enumerate(predsRel):
        if p < 0.5:
            numRelated += 1
            X_holdout_rel.append(X_holdout[i])
        else:
            numUnrelated += 1
    print(numRelated)
    print(numUnrelated)

    predsType = modelType.predict(np.asarray(X_holdout_rel))
    ind = 0
    predicted_holdout = []
    for i, p in enumerate(predsRel):
        prediction = 0
        if p < 0.5:
            prediction = np.argmax(predsType[ind])
            ind += 1
        else:
            prediction = 3
        predicted_holdout.append(LABELS[prediction])

    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual, predicted_holdout)
    print("")
    print("")



    # Run on competition dataset
    predsRel = modelRel.predict(X_competition)
    X_competition_type = []
    numRelated = 0
    numUnrelated = 0
    for i, p in enumerate(predsRel):
        if p < 0.5:
            numRelated += 1
            X_competition_type.append(X_competition[i])
        else:
            numUnrelated += 1

    predsType = modelType.predict(np.asarray(X_competition_type))
    ind = 0
    predicted_competition = []
    for i, p in enumerate(predsRel):
        prediction = 0
        if p < 0.5:
            prediction = np.argmax(predsType[ind])
            ind += 1
        else:
            prediction = 3
        predicted_competition.append(LABELS[prediction])

    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted_competition)

