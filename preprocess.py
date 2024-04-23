## Methods to preprocess the data go here.
import pandas as pd
import json
import re
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
        try:
            # Extract the text and label from the parts
            text = ','.join(parts[:-1])  # Joining in case the text contains commas
            label = int(parts[-1])
            # Append the text and label to the data list
            data.append((text, label))
        except ValueError:
            # Skip the row if ValueError occurs
            continue

    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])

    # Split the DataFrame into features (X) and labels (y)
    X = df['text']  # Features (text)
    y = df['label']  # Labels (0 for human-written, 1 for AI-written)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test

def filterAndConvertToCSV(input_filepath, output_filepath):
    data = []
    with open(input_filepath, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            # Extract human answer if available
            human_answers = json_data.get("human_answers", [])
            if human_answers:
                human_answer = human_answers[0].replace("\n", "")
                human_answer = human_answers[0].replace(" '", "'")
                human_answer = re.sub(r"\b(do|does|did|wo|ca|sha|would|could|should|must|might|may|is|are|was|were) n't\b", r"\1n't", human_answer)
                if human_answer.startswith('> '):
                    human_answer = human_answer[2:]  # Remove the first two characters ('> ')
                data.append({"Data": human_answer, "Labels": 0})
            # Extract AI answer if available
            ai_answers = json_data.get("chatgpt_answers", [])
            if ai_answers:
                ai_answer = ai_answers[0].replace("\n", "")
                ai_answer = ai_answers[0].replace(" '", "'")
                ai_answer = re.sub(r"\b(do|does|did|wo|ca|sha|would|could|should|must|might|may|is|are|was|were) n't\b", r"\1n't", ai_answer)
                data.append({"Data": ai_answer, "Labels": 1})

    # Convert the filtered data to a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_filepath, index=False)