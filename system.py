## Code for the system goes here
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf_vectorizer = TfidfVectorizer()
classifier = LogisticRegression()