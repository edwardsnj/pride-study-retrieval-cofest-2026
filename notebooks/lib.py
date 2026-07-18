BASEURL = "https://edwardslab.bmcb.georgetown.edu/~nedwards/dropbox/6ItUS2tEdC/"
GITHUB = "https://raw.githubusercontent.com/EdwardsLabProjects/pride-study-retrieval-cofest-2026/refs/heads/main/data/"

import os, os.path, subprocess

VERSION='1.0.33'

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
      nbgsel = min(int(round(len(seeds)*bgsize)), len(bg))

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
      num_bg_train_samples = int(round(nbgsel*(1-test_size)))
      num_test_samples = len(pos_test_accs)
      num_bg_test_samples = nbgsel-num_bg_train_samples

      n_extra = max(0, nbgsel - len(neg_seeds))
      selbg = neg_seeds + list(random.sample(list(set(bg)-set(neg_seeds)), n_extra))
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
      fit_accessions = [acc for acc, y_val in zip(train_accessions, train_y) if y_val == 1]
    else:
      fit_accessions = train_accessions

    # Prepare true case training data for TF-IDF fitting
    # Ensure the order of texts matches the order of true_train_accessions for correct indexing
    md_fit_cases = md_dataframe[md_dataframe['prideacc'].isin(fit_accessions)].set_index('prideacc').loc[fit_accessions]
    texts_fit_cases = md_fit_cases['text']

    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(**kwargs)

    # Fit TfidfVectorizer ONLY on the true cases of the training data
    tfidf_vectorizer.fit(texts_fit_cases)

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

from sklearn.metrics.pairwise import cosine_similarity

def select_by_embedding_proximity(tp, emb, n=1000):
    tp_accs = [acc for acc in tp if acc in emb.columns]
    avg_emb = emb[tp_accs].mean(axis=1).values.reshape(1, -1)
    sims = cosine_similarity(avg_emb, emb.values.T)[0]
    sim_series = pd.Series(sims, index=emb.columns)
    non_tp_sims = sim_series.drop(labels=tp_accs, errors='ignore')
    top_non_tp = non_tp_sims.nlargest(max(0, n - len(tp_accs))).index.tolist()
    return tp_accs + top_non_tp, avg_emb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle

def train_document_classifier(embeddings, tfidf, train_acc, train_y, test_acc, test_y, use_embed=True, use_tfidf=True, **kwargs):

    # Use np.hstack to concatenate features horizontally
    train_values = []
    if use_embed:
        train_values.append(embeddings[train_acc].values.T)
    if use_tfidf:
        train_values.append(tfidf[train_acc].values.T)
    train = np.hstack(train_values)
    if len(test_acc)>0:
        have_test = True
        # Use np.hstack for test features as well
        test_values = []
        if use_embed:
            test_values.append(embeddings[test_acc].values.T)
        if use_tfidf:
            test_values.append(tfidf[test_acc].values.T)
        test = np.hstack(test_values)

    # 4. Shuffle the datasets so labels are randomized (not all 1s followed by all 0s)
    X_train, y_train = shuffle(train, train_y)
    if have_test:
        X_test, y_test = shuffle(test, test_y)

    print(f"Training data shape: {X_train.shape} (Positives: {sum((y==1) for y in y_train)}, Negatives: {sum((y==0) for y in y_train)})")
    if have_test:
        print(f"Testing data shape: {X_test.shape} (Positives: {sum((y==1) for y in y_test)}, Negatives: {sum((y==0) for y in y_test)})")

    # 5. Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, **kwargs)
    model.fit(X_train, y_train)

    # 6. Evaluate the model on the withheld test set
    if not have_test:
        return model

    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    # target_names let us interpret the 1s and 0s easily in the terminal
    print(classification_report(y_test, y_pred, target_names=["Background (0)", "Seed-like (1)"]))

    return model

def top_features(logreg_model,tfidf_model,nembed=0,use_embed=True,use_tfidf=True,**kwargs):

    parts = []
    significant_embedding_coeffs = 0
    non_zero_tfidf_coeffs = 0

    if use_embed:
        embedding_coefficients = logreg_model.coef_[0][:nembed]
        significant_embedding_coeffs = np.sum(embedding_coefficients != 0)
        parts.append(pd.DataFrame({
            'Feature': [f'embed_{i}' for i in range(nembed)],
            'Coefficient': embedding_coefficients
        }))

    if use_tfidf:
        tfidf_feature_names = tfidf_model.get_feature_names_out()
        tfidf_coefficients = logreg_model.coef_[0][nembed:]
        non_zero_tfidf_coeffs = np.sum(tfidf_coefficients != 0)
        parts.append(pd.DataFrame({
            'Feature': tfidf_feature_names,
            'Coefficient': tfidf_coefficients
        }))

    all_features = pd.concat(parts, ignore_index=True)
    all_features['Abs_Coefficient'] = all_features['Coefficient'].abs()
    most_important_features = all_features[all_features['Coefficient'] != 0].sort_values(by='Abs_Coefficient', ascending=False)

    return significant_embedding_coeffs, non_zero_tfidf_coeffs, most_important_features.drop(columns=['Abs_Coefficient'])

from tqdm import tqdm

def score_all_studies(model, emb, md, tfidf_vectorizer, train_accessions, tp, tn, use_embed=True, use_tfidf=True, batch_size=500):
    md_indexed = md.set_index('prideacc')
    allacc = [acc for acc in emb.columns if acc in md_indexed.index]
    batches = [allacc[i:i+batch_size] for i in range(0, len(allacc), batch_size)]

    all_probs = []
    for batch in tqdm(batches, desc="Scoring studies", ascii=True):
        feature_parts = []
        if use_embed:
            feature_parts.append(emb[batch].values.T)
        if use_tfidf:
            texts = md_indexed.loc[batch, 'text']
            feature_parts.append(tfidf_vectorizer.transform(texts).toarray())
        all_probs.extend(model.predict_proba(np.hstack(feature_parts))[:, 1])

    def extract_title(text):
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                return line[2:]
        return ''

    train_set, tp_set, tn_set = set(train_accessions), set(tp), set(tn)
    results = pd.DataFrame({
        'prideacc': allacc,
        'title': [extract_title(t) for t in md_indexed.loc[allacc, 'text']],
        'probability': [round(p, 4) for p in all_probs],
        'in_training': [acc in train_set for acc in allacc],
        'true_positive': [acc in tp_set for acc in allacc],
        'true_negative': [acc in tn_set for acc in allacc],
    })
    return results.sort_values('probability', ascending=False).reset_index(drop=True)

import re

def show_top_feature_examples(top_features_df, md_dataframe, n_features=10, n_examples=5):
    for _, row in top_features_df.head(n_features).iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        print(f"\n=== '{feature}' (coef: {coef:+.4f}) ===")
        pattern = re.compile(re.escape(feature), re.IGNORECASE)
        examples = []
        for _, study in md_dataframe.iterrows():
            text = study['text']
            if not pattern.search(text):
                continue
            fragments = re.split(r'\n\n+|(?<=[.!?])\s+', text)
            match = next((f for f in fragments if pattern.search(f)), None)
            if match:
                examples.append((study['prideacc'], match.strip()))
            if len(examples) >= n_examples:
                break
        for acc, sentence in examples:
            print(f"  {acc}: {sentence}")

print(f"Version: {VERSION}")

   