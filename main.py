from model import model
from logic import prepare_files
import numpy as np
from sklearn.model_selection import train_test_split

train_x, train_y = prepare_files('cifar-10-batches-py/data_batch_{}')
test_x, test_y = prepare_files('cifar-10-batches-py/test_batch')
print('\nData obtained')

x = np.concatenate((train_x, test_x))
y = np.concatenate((train_y, test_y))
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.1)

model = model()
# model.fit(train_x, train_y, batch_size=128, n_epoch=14)
model.load('cifar-10.model')
# model.save('cifar-10.model')
accuracy = model.evaluate(test_x, test_y, batch_size=128)
print('Accuracy: ', accuracy)
