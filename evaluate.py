import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

## Methods for evaluating the system output go here

## accuracy
def true_positive(df):
  return len(df[(df['pred'] == 1 ) & (df['gt'] == 1)].index)

def true_negative(df):
  return len(df[(df['pred'] == 0 ) & (df['gt'] == 0)].index)

TP = true_positive(data)
P_gt_total = gt.get_group(1).shape[0]
acc_P = TP / P_gt_total

TN = true_negative(data)
N_gt_total = gt.get_group(0).shape[0]
acc_N = TN / N_gt_total

balanced_acc = (acc_P + acc_N)/2

print("TNR: ", acc_N)
print("TPR: ", acc_P)
print("Balanced ACC: ", balanced_acc)

##confusion matrix

# Sample data
active_account_data = data[data['type'] != "U"]
true_labels = np.array(active_account_data['gt'])
predicted_labels = np.array(active_account_data['pred'])

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, normalize = 'all')

# Define custom colors
class_colors = np.array([['palegreen', 'lightcoral'],
                        ['lightcoral', 'palegreen']])

# Plot the confusion matrix without heatmap
fig, ax = plt.subplots()

# Loop through the confusion matrix and color the boxes
for i in range(len(cm)):
    for j in range(len(cm[0])):
        color = class_colors[j, i]  # Reverse the order for rows
        rect = plt.Rectangle((j, len(cm) - 1 - i), 1, 1, fill=True, color=color, edgecolor='black')  # Reverse the order for rows
        ax.add_patch(rect)
        ax.text(j + 0.5, len(cm) - 1 - i + 0.5, f"{cm[i, j]:.2f}%", va='center', ha='center', color='black', fontsize=15)  # Reverse the order for rows

# Set the x and y limits
ax.set_xlim(0, len(cm))
ax.set_ylim(0, len(cm))

# Set x-axis and y-axis labels
ax.set_xticks(np.arange(len(cm[0])) + 0.5, minor=False)
ax.set_yticks(np.arange(len(cm)) + 0.5, minor=False)
ax.set_xticklabels(['0', '1'])
ax.set_yticklabels(['1', '0'])  # Reverse the order for y-axis

# Add labels and title
plt.xlabel("Predicted label", fontsize = 16)
plt.ylabel("True label", fontsize = 16)
plt.figure(figsize=(3,2))

plt.show()