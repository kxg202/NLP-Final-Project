## Will call methods from other files, this is where the program is run.
from preprocess import *
from system import *
from evaluate import *

#x_train, x_test, y_train, y_test = preprocessCSV("./datasets/reddit_filtered_dataset.csv")
x_train, x_test, y_train, y_test = preprocessCSV("./datasets/reddit_eli5_filtered.csv")
#filterAndConvertToCSV("./datasets/reddit_eli5.jsonl", "./datasets/reddit_eli5_filtered.csv")
y_pred, accuracy, precision, recall, f1 = train_and_predict(x_train, x_test, y_train, y_test)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize = 'all')
#visualize_confusion_matrix(cm)
visualize_confusion_matrix(cm, ['0', '1'], ['1', '0'])