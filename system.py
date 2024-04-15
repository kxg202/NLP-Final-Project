## Code for the system goes here
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_and_predict(X_train, X_test, Y_train, Y_test):
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    tfidf_vectorizer = TfidfVectorizer(ngram_range(1,2))
    classifier = LogisticRegression()
    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)
    return y_pred