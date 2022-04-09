import sys
from this import s
import numpy as np

import math
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, sentiment_intensity_features, cosine_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
from csv import DictReader
import pandas


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
    X_cosine = gen_or_load_feats(cosine_features, h, b, "features/cosine."+name+".npy")
    X_sent = gen_or_load_feats(sentiment_intensity_features, h, b, "features/sent."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_cosine, X_sent]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=15)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    h, b = [], []
    for stance in competition_dataset.stances:
        h.append(stance['Headline'])
        b.append(stance['Body ID'])

    answers = {'Headline': h, 'Body ID': b, 'Stance': []}

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score_rel = 0
    best_fold_rel = None
    best_score_type = 0
    best_fold_type = None


    # Classifier for each fold
    t = 0
    for fold in fold_stances:
        '''if t > 0:
            break
        t = 1'''
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

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

        min_samp_rel = math.floor(0.01*len(y_train_rel))
        min_samp_type = math.floor(0.005*len(y_train_type))
        clfRel = GradientBoostingClassifier(n_estimators=65, min_samples_split=min_samp_rel, learning_rate=0.2, min_samples_leaf=50, verbose=True)
        clfType = GradientBoostingClassifier(n_estimators=65, min_samples_split=min_samp_type, learning_rate=0.2, min_samples_leaf=50, verbose=True)

        clfRel.fit(X_train, y_train_rel)
        clfType.fit(X_train_type, y_train_type)

        predicted_rel = [LABELS[int(a)] for a in clfRel.predict(X_test)]
        actual_rel = [LABELS[int(a)] for a in y_test_rel]
        fold_score_rel, _ = score_submission(actual_rel, predicted_rel)
        max_fold_score_rel, _ = score_submission(actual_rel, actual_rel)
        score_rel = fold_score_rel / max_fold_score_rel
        print("Score for fold " + str(fold) + " was - " + str(score_rel))
        if score_rel > best_score_rel:
            best_score_rel = score_rel
            best_fold_rel = clfRel

        predicted_type = [LABELS[int(a)] for a in clfRel.predict(X_test_type)]
        actual_type = [LABELS[int(a)] for a in y_test_type]
        fold_score_type, _ = score_submission(actual_type, predicted_type)
        max_fold_score_type, _ = score_submission(actual_type, actual_type)
        score_type = fold_score_type / max_fold_score_type
        print("Score for fold " + str(fold) + " was - " + str(score_type))
        if score_type > best_score_type:
            best_score_type = score_type
            best_fold_type = clfType

    # Run on Holdout set and report the final score on the holdout set
    # predicted = [LABELS[np.argmax(a, axis = 0)] for a in modelRel.predict(X_holdout)]
    predsRel = best_fold_rel.predict(X_holdout)
    X_holdout_rel = []
    for i, p in enumerate(predsRel):
        if p == 0:
            X_holdout_rel.append(X_holdout[i])

    predsType = best_fold_type.predict(np.asarray(X_holdout_rel))
    ind = 0
    predicted_holdout = []
    for i, p in enumerate(predsRel):
        prediction = 0
        if p == 0:
            prediction = predsType[ind]
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
    predsRel = best_fold_rel.predict(X_competition)
    X_competition_type = []
    numRelated = 0
    numUnrelated = 0
    for i, p in enumerate(predsRel):
        if p == 0:
            numRelated += 1
            X_competition_type.append(X_competition[i])
        else:
            numUnrelated += 1

    predsType = best_fold_type.predict(np.asarray(X_competition_type))
    ind = 0
    predicted_competition = []
    for i, p in enumerate(predsRel):
        prediction = 0
        if p == 0:
            prediction = predsType[ind]
            ind += 1
        else:
            prediction = 3
        predicted_competition.append(LABELS[prediction])
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted_competition)

    answers["Stance"] = predicted_competition
    answers = pandas.DataFrame(answers)
    answers.to_csv('answer.csv', index=False, encoding='utf-8')