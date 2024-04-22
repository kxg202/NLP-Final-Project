## Will call methods from other files, this is where the program is run.
from preprocess import *
from system import *
from evaluate import *

preprocess()
train()
evaluate()