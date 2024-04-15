## Methods to preprocess the data go here.
from sklearn.model_selection import train_test_split

def preprocessCSV(filepath):
    f = open("datasets" + filepath)
    return x_train, x_test, y_train, y_test

def preprocessJSON(filepath):
    f = open("datasets" + filepath)
    return x_train, x_test, y_train, y_test

def preprocess(filepath):
    file_extension = filepath.split(".")[-1]
    extensionMap = {
        "json": preprocessJSON,
        "csv": preprocessCSV
    }
    if file_extension in extensionMap:
        return extensionMap[file_extension](filepath)
    else:
        print("Unsupported file extension:", file_extension)
    