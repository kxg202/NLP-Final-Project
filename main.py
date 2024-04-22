## Will call methods from other files, this is where the program is run.
from preprocess import *
from system import *
from evaluate import *

x_train, x_test, y_train, y_test = preprocess("./datasets/reddit_filtered_dataset.csv")
print(x_train)
#train()
#evaluate()