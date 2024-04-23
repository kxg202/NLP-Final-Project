## Will call methods from other files, this is where the program is run.
from preprocess import *
from system import *

x_train, x_test, y_train, y_test = preprocessCSV("./datasets/reddit_filtered_dataset.csv")
#x_train, x_test, y_train, y_test = preprocessCSV("./datasets/reddit_eli5_filtered.csv")
#filterAndConvertToCSV("./datasets/reddit_eli5.jsonl", "./datasets/reddit_eli5_filtered.csv")
y_pred, accuracy, precision, recall, f1 = train_and_predict(x_train, x_test, y_train, y_test)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")