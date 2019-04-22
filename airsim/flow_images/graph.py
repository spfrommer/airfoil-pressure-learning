import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

import airsim.dirs as dirs
import matplotlib.pyplot as plt

f = open(dirs.out_path('training', 'data.log'), 'r')

epochs = []
train_errors = []
test_errors = []

for line in f:
    parsed_item_list = line.split(', ')
    values = [float(x) for x in parsed_item_list]
    epochs.append(values[0])
    train_errors.append(values[1])
    test_errors.append(values[2])

plt.plot(epochs, train_errors, label='Train Error')
plt.plot(epochs, test_errors, label='Test Error')
plt.title('Error per Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean-Squared Error')
plt.legend(loc='upper right')
plt.show()
    
