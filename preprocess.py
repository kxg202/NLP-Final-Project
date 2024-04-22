## Methods to preprocess the data go here.
import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO

def preprocessCSV(filepath, test_size=0.2, random_state=None):
    # Read the CSV file into a pandas DataFrame
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Initialize lists to store features and labels
    data = []

    # Iterate over each line in the CSV file
    for line in lines:
        # Split the line by comma
        parts = line.strip().split(',')

        # Extract the text and label from the parts
        text = ','.join(parts[:-1])  # Joining in case the text contains commas
        label = int(parts[-1])

        # Append the text and label to the data list
        data.append((text, label))

    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])

    # Split the DataFrame into features (X) and labels (y)
    X = df['text']  # Features (text)
    y = df['label']  # Labels (0 for human-written, 1 for AI-written)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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
    