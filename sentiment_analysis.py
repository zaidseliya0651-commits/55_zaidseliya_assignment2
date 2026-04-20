"""
Assignment 2 - Sentiment Analysis on ChatGPT Tweets
Course: Data Analytics and Visualisation (CSC601)
Class: TE AIDS (2025-26), Semester VI
Institution: Rizvi College of Engineering
"""

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv("tweets_dataset.csv")
print(f"Total tweets loaded: {len(df)}")
print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
print("\nSample tweets:")
print(df[['tweet', 'sentiment']].head(5).to_string(index=False))


# ============================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Text Preprocessing")
print("=" * 60)

def preprocess_text(text):
    """
    Preprocess tweet text:
    - Convert to lowercase
    - Remove URLs
    - Remove mentions (@user)
    - Remove hashtags symbol (keep word)
    - Remove special characters and punctuation
    - Remove extra whitespace
    - Remove stopwords (manual list)
    """
    stopwords = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
        'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'shall',
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'into',
        'through', 'during', 'to', 'from', 'in', 'out', 'on', 'off',
        'so', 'than', 'too', 'very', 'just', 'also', 'then', 'when',
        'up', 'down', 'no', 'not', 'can', 'its'
    ])

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol
    text = re.sub(r'#', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return ' '.join(tokens)

df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)
print("Preprocessing complete!")
print("\nBefore preprocessing:", df['tweet'][0])
print("After preprocessing: ", df['cleaned_tweet'][0])


# ============================================================
# STEP 3: ENCODE LABELS & SPLIT DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Encoding Labels & Splitting Dataset (80/20)")
print("=" * 60)

label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
df['label'] = df['sentiment'].map(label_map)

X = df['cleaned_tweet']
y = df['label']

# Fixed random_state for reproducibility | 80 train, 20 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set size : {len(X_train)} tweets")
print(f"Testing  set size : {len(X_test)} tweets")
print(f"\nTraining label distribution:\n{y_train.value_counts().rename({2:'positive',1:'neutral',0:'negative'})}")
print(f"\nTesting  label distribution:\n{y_test.value_counts().rename({2:'positive',1:'neutral',0:'negative'})}")


# ============================================================
# STEP 4: TF-IDF VECTORIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: TF-IDF Vectorization")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=500,      # Top 500 features
    ngram_range=(1, 2),    # Unigrams + Bigrams
    sublinear_tf=True      # Apply log normalization
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"Vocabulary size   : {len(tfidf.vocabulary_)}")
print(f"Training matrix   : {X_train_tfidf.shape}")
print(f"Testing  matrix   : {X_test_tfidf.shape}")


# ============================================================
# STEP 5: TRAIN CLASSIFIERS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training Classifiers")
print("=" * 60)

classifiers = {
    "Naive Bayes":       MultinomialNB(alpha=1.0),
    "SVM":               LinearSVC(C=1.0, max_iter=2000, random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42)
}

results = {}
label_names = ['negative', 'neutral', 'positive']

for name, clf in classifiers.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score   (y_test, y_pred, average='weighted', zero_division=0)
    accuracy  = accuracy_score (y_test, y_pred)

    results[name] = {
        'model':     clf,
        'y_pred':    y_pred,
        'precision': precision,
        'recall':    recall,
        'accuracy':  accuracy
    }
    print(f"\n--- {name} ---")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=label_names,
                                 zero_division=0))


# ============================================================
# STEP 6: COMPARISON TABLE
# ============================================================
print("=" * 60)
print("STEP 6: Classifier Comparison")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Classifier': list(results.keys()),
    'Accuracy':   [results[k]['accuracy']  for k in results],
    'Precision':  [results[k]['precision'] for k in results],
    'Recall':     [results[k]['recall']    for k in results]
})
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
print(comparison_df.to_string(index=False))

best = comparison_df.iloc[0]['Classifier']
print(f"\n✅ Best Classifier: {best}")


# ============================================================
# STEP 7: VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Generating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Sentiment Analysis on ChatGPT Tweets — Model Results',
             fontsize=16, fontweight='bold', y=1.01)

colors_bar = ['#4C72B0', '#DD8452', '#55A868']

# --- Row 1: Confusion Matrices ---
for idx, (name, res) in enumerate(results.items()):
    ax = axes[0][idx]
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_title(f'Confusion Matrix\n{name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# --- Row 2 Left: Precision Comparison Bar Chart ---
ax2 = axes[1][0]
bars = ax2.bar(comparison_df['Classifier'], comparison_df['Precision'],
               color=colors_bar, edgecolor='black', width=0.5)
ax2.set_title('Precision Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision (Weighted)')
ax2.set_ylim(0, 1.1)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
ax2.tick_params(axis='x', rotation=15)

# --- Row 2 Centre: Recall Comparison Bar Chart ---
ax3 = axes[1][1]
bars2 = ax3.bar(comparison_df['Classifier'], comparison_df['Recall'],
                color=colors_bar, edgecolor='black', width=0.5)
ax3.set_title('Recall Comparison', fontsize=12, fontweight='bold')
ax3.set_ylabel('Recall (Weighted)')
ax3.set_ylim(0, 1.1)
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
ax3.tick_params(axis='x', rotation=15)

# --- Row 2 Right: Accuracy Comparison Bar Chart ---
ax4 = axes[1][2]
bars3 = ax4.bar(comparison_df['Classifier'], comparison_df['Accuracy'],
                color=colors_bar, edgecolor='black', width=0.5)
ax4.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy')
ax4.set_ylim(0, 1.1)
for bar in bars3:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
ax4.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('results_chart.png', dpi=150, bbox_inches='tight')
print("Saved: results_chart.png")
plt.show()


# ============================================================
# STEP 8: SENTIMENT DISTRIBUTION PIE CHART
# ============================================================
fig2, ax5 = plt.subplots(figsize=(7, 7))
counts = df['sentiment'].value_counts()
colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
wedges, texts, autotexts = ax5.pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=colors_pie, startangle=140,
    textprops={'fontsize': 13}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
ax5.set_title('Tweet Sentiment Distribution (100 Tweets)',
              fontsize=14, fontweight='bold', pad=20)
plt.savefig('sentiment_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: sentiment_distribution.png")
plt.show()


# ============================================================
# STEP 9: SAVE RESULTS TO CSV
# ============================================================
comparison_df.to_csv('model_results.csv', index=False)
print("\nSaved: model_results.csv")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"Best Classifier : {best}")
print(f"Precision       : {results[best]['precision']:.4f}")
print(f"Recall          : {results[best]['recall']:.4f}")
print(f"Accuracy        : {results[best]['accuracy']:.4f}")
