BASEURL = "https://edwardslab.bmcb.georgetown.edu/~nedwards/dropbox/6ItUS2tEdC/"
GITHUB = "https://raw.githubusercontent.com/EdwardsLabProjects/pride-study-retrieval-cofest-2026/refs/heads/main/data/"

import os, os.path, subprocess
import pandas

def download_embeddings(model="openai-3-small"):
    # files...
    csvfile = f"pride-embeddings-{model}.csv"
    fthfile = f"pride-embeddings-{model}.fth"
    for f in [csvfile, fthfile]:
      if not os.path.exists(f):
        subprocess.run(["wget", BASEURL+f])
    return csvfile, fthfile

def download_knownstudies():
    trueposfile = "truepos.txt"
    truenegfile = "trueneg.txt"
    for f in [trueposfile, truenegfile]:
      if not os.path.exists(f):
        subprocess.run(["wget", GITHUB+f])
    return trueposfile,truenegfile

import numpy as np
import pandas as pd
import random

def set_random_seed(state=None):
    if not state:
        state = random.randint(1,10000000)

    print(f"Using random seed: {state}")

    # Seeds all scikit-learn functions that default to random_state=None
    np.random.seed(state)
    random.seed(state)

def embeddings(model="openai-3-small"):
    csvfile,fthfile = download_embeddings(model)
    emb = pd.read_feather(fthfile)
    md = pd.read_csv(csvfile)
    return md,emb

def knownstudies():
    trueposfile,truenegfile = download_knownstudies()
    tp = set(open(trueposfile).read().split())
    tn = set(open(truenegfile).read().split())
    assert len(set(tp) & set(tn)) == 0, "TP and TN should not intersect!"
    return tp,tn

from sklearn.model_selection import train_test_split

def split_train_test(allacc, seeds, neg_seeds, test_size=0.2, bgsize=25):
      seeds = list(set(seeds)&set(allacc))
      neg_seeds = list(set(neg_seeds)&set(allacc))
      bg = list(set(allacc)-set(seeds))
      nbgsel = int(round(len(seeds)*bgsize))
      if len(bg) < len(neg_seeds):
        raise ValueError("Not enough background embeddings to create selected background.")
      if len(bg) < nbgsel:
        raise ValueError("Not enough background embeddings to create selected background.")

      if test_size > 0.0:
        # 1. Split the original seed set
        pos_train_accs, pos_test_accs = train_test_split(
            seeds,
            test_size=test_size
        )
        have_test = True
      else:
        pos_train_accs = list(seeds)
        pos_test_accs = []
        have_test = False

      selected_accessions = list(seeds)
      train_accessions = list(pos_train_accs)
      test_accessions = list(pos_test_accs)

      num_train_samples = len(pos_train_accs)
      num_bg_train_samples = int(round(num_train_samples*bgsize))
      num_test_samples = len(pos_test_accs)
      num_bg_test_samples = nbgsel-num_bg_train_samples

      selbg = neg_seeds + list(random.sample(list(set(bg)-set(neg_seeds)),
                                             nbgsel-len(neg_seeds)))
      random.shuffle(selbg)
      seltrainbg = selbg[:num_bg_train_samples]
      seltestbg = selbg[num_bg_train_samples:]

      selected_accessions += selbg
      train_accessions += seltrainbg
      test_accessions += seltestbg
      train_y = [1]*num_train_samples + [0]*num_bg_train_samples
      test_y = [1]*num_test_samples + [0]*num_bg_test_samples

      return train_accessions, train_y, test_accessions, test_y

from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(md_dataframe, train_accessions, train_y, test_accessions, positive_only=False,**kwargs):
    # Filter training accessions to include only 'true cases' (where train_y is 1)
    if positive_only:
      true_train_accessions = [acc for acc, y_val in zip(train_accessions, train_y) if y_val == 1]
    else:
      true_train_accessions = train_accessions

    # Prepare true case training data for TF-IDF fitting
    # Ensure the order of texts matches the order of true_train_accessions for correct indexing
    md_train_true_cases = md_dataframe[md_dataframe['prideacc'].isin(true_train_accessions)].set_index('prideacc').loc[true_train_accessions]
    train_texts_true_cases = md_train_true_cases['text']

    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(**kwargs)

    # Fit TfidfVectorizer ONLY on the true cases of the training data
    tfidf_vectorizer.fit(train_texts_true_cases)

    # Prepare ALL training data for transformation
    md_train_all = md_dataframe[md_dataframe['prideacc'].isin(train_accessions)].set_index('prideacc').loc[train_accessions]
    train_texts_all = md_train_all['text']

    # Transform ALL training data using the fitted vectorizer
    tfidf_train_matrix = tfidf_vectorizer.transform(train_texts_all)

    # Create a DataFrame for training TF-IDF values
    tfidf_df_train = pd.DataFrame(
        tfidf_train_matrix.toarray(),
        index=train_accessions, # Index by pride accessions
        columns=tfidf_vectorizer.get_feature_names_out()
    )

    # Prepare testing data for TF-IDF transformation
    md_test = md_dataframe[md_dataframe['prideacc'].isin(test_accessions)].set_index('prideacc').loc[test_accessions]
    test_texts = md_test['text']

    # Apply the fitted TF-IDF model to test data (transform only)
    tfidf_test_matrix = tfidf_vectorizer.transform(test_texts)

    # Create a DataFrame for testing TF-IDF values
    tfidf_df_test = pd.DataFrame(
        tfidf_test_matrix.toarray(),
        index=test_accessions, # Index by pride accessions
        columns=tfidf_vectorizer.get_feature_names_out()
    )

    tfidf_df = pd.concat([tfidf_df_train, tfidf_df_test]).T

    return tfidf_df, tfidf_vectorizer

VERSION='1.0.3'
print(f"Version: {VERSION}")

   