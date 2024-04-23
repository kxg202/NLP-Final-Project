## Code for the system goes here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_predict(X_train, X_test, y_train, y_test):
    # Initialize and fit the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Initialize the logistic regression classifier
    classifier = LogisticRegression()
    
    # Train the classifier
    classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_tfidf)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Return predicted labels and evaluation metrics
    return y_pred, accuracy, precision, recall, f1
    
