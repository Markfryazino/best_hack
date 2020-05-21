import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def prepare_features(train, test, scale=True):
    train, test = encode_str(train, test)
    data = pd.concat([train, test], sort=False)
    data = tfidf_desc(data)
    data.drop(['Vit_D_µg', 'Vit_A_RAE', 'Folate_Tot_(µg)'], axis=1, inplace=True)
    data = handle_floats(data)
    data = polynomial_features(data)
    data = drop_excess(data)
    train, test, scaler = split_back(data, scale)
    return train, test, scaler


def prepare_reduced_features(train, test, scale=True):
    data = pd.concat([train, test], sort=False)
    data = tfidf_desc(data)
    data = handle_floats(data)
    data = polynomial_features(data, True)
    data.drop('Shrt_Desc', axis=1, inplace=True)
    train, test, scaler = split_back(data, scale)
    return train, test, scaler


def handle_desc(x, param):
    if (x != x) and (param == 0):
        return np.NaN
    elif x != x:
        return 'nan'
    tup = x.lower().replace(',', '').split(' ')[:2]
    if param == 0:
        return float(tup[0])
    elif param == 1:
        return tup[1]
    else:
        return ' '.join(tup)


def handle_shrt_desc(x):
    new = ''
    for el in x.lower():
        if el.isalpha():
            new += el
        else:
            new += ' '
    return new


def encode_str(train, test):
    for col in ['GmWt_Desc1', 'GmWt_Desc2']:
        train[col + '_first'] = train[col].apply(lambda x: handle_desc(x, 0))
        test[col + '_first'] = test[col].apply(lambda x: handle_desc(x, 0))

        train[col + '_second'] = train[col].apply(lambda x: handle_desc(x, 1))
        test[col + '_second'] = test[col].apply(lambda x: handle_desc(x, 1))

        train[col + '_pair'] = train[col].apply(lambda x: handle_desc(x, 2))
        test[col + '_pair'] = test[col].apply(lambda x: handle_desc(x, 2))

        cv = KFold(n_splits=5, shuffle=True)

        for cur_col in [col + '_second', col + '_pair']:
            globalmean = train['Energ_Kcal'].mean()
            train[cur_col + '_encoded'] = 0.
            for train_idx, test_idx in cv.split(train):
                mapping = train.loc[train_idx].groupby(cur_col)['Energ_Kcal'].mean()
                train[cur_col + '_encoded'][test_idx] = train[cur_col][test_idx].map(mapping)
            train[cur_col + '_encoded'].fillna(globalmean, inplace=True)
            mapping = train.groupby(cur_col)['Energ_Kcal'].mean()
            test[cur_col + '_encoded'] = test[cur_col].map(mapping)
            test[cur_col + '_encoded'].fillna(globalmean, inplace=True)
    return train, test


def tfidf_desc(data):
    pd_corpus = data['Shrt_Desc'].apply(handle_shrt_desc)
    corpus = list(pd_corpus.values)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    nmf = NMF(n_components=5)
    word_nmf = nmf.fit_transform(tfidf)
    word_cols = ['word_TF-IDF-' + str(i) for i in range(5)]
    return pd.concat([data.reset_index(drop=True), pd.DataFrame(word_nmf, columns=word_cols)],
                     axis=1, sort=False)


def handle_floats(data):
    data['Energ_Kcal'].fillna(-1., inplace=True)
    data['NaN_number'] = data.isnull().sum(axis=1)

    float_cols = list(set(data.columns) - {'GmWt_Desc2_pair', 'GmWt_Desc2_second', 'GmWt_Desc1_pair',
                                           'GmWt_Desc1', 'GmWt_Desc2', 'Shrt_Desc', 'Energ_Kcal',
                                           'GmWt_Desc1_second'})
    for col in float_cols:
        data[col].fillna(data[col].median(), inplace=True)
        data[col] = StandardScaler().fit_transform(data[col].values.reshape(-1, 1))
    return data


def drop_excess(data):
    str_cols = ['GmWt_Desc1', 'GmWt_Desc2', 'Shrt_Desc', 'GmWt_Desc2_pair', 'GmWt_Desc2_second',
                'GmWt_Desc1_pair', 'GmWt_Desc1_second']
    data.drop(str_cols, axis=1, inplace=True)
    return data


def split_back(data, scale):
    train = data[data['Energ_Kcal'] != -1]
    test = data[data['Energ_Kcal'] == -1]
    test.drop('Energ_Kcal', axis=1, inplace=True)

    if scale:
        scaler = StandardScaler()
        scaler.fit(train['Energ_Kcal'].values.reshape(-1, 1))
        train['Energ_Kcal'] = scaler.transform(train['Energ_Kcal'].values.reshape(-1, 1))
    else:
        scaler = None
    return train, test, scaler


def polynomial_features(data, use_subset=False):
    if not use_subset:
        important = ['Water_(g)', 'Lipid_Tot_(g)', 'FA_Mono_(g)', 'FA_Sat_(g)', 'FA_Poly_(g)',
                     'Carbohydrt_(g)', 'GmWt_1', 'Vit_E_(mg)', 'Sugar_Tot_(g)']
    else:
        important = ['Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Protein_(g)']
    mapping = {'x' + str(i): important[i] for i in range(len(important))}
    poly = PolynomialFeatures(include_bias=False)
    the = poly.fit_transform(data[important])
    feats = poly.get_feature_names()
    for i in range(len(feats)):
        for key, val in mapping.items():
            feats[i] = feats[i].replace(key, val)
    the = pd.DataFrame(the, columns=feats)
    return pd.concat([data.reset_index(drop=True), the], axis=1, sort=False)
