import numpy as np
import matplotlib.pyplot as plt
from numba.cuda.printimpl import print_item
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

#exercitiul 1
heights = np.array([160, 165, 155, 172, 175, 180, 177, 190])
labels = np.array(['F' ,'F', 'F', 'F', 'B', 'B', 'B', 'B'])

bins = [150, 161, 171, 181, 191]
x_to_bins = np.digitize(heights, bins)

idx_interval = 3
mask_interval = (x_to_bins == 3)
cnt = mask_interval.sum()

f = ((labels == 'F') & mask_interval).sum()
b = ((labels == 'B') & mask_interval).sum()

pI3_F = f / cnt
pI3_B = b / cnt

pF = (labels == 'F').mean()
pB = (labels == 'B').mean()

pI3 = pI3_F * pF + pI3_B * pB

pF_I3 = (pI3_F * pF) / pI3
print(pF_I3)

#exercitiul 2

num_bins = 5
bins = np.linspace(0, 255, num_bins)

def value_to_bins(X, bins):
    return np.digitize(X, bins)

arrayTest = np.loadtxt('data/test_images.txt')
arrayTestY = np.loadtxt('data/test_labels.txt')
arrayTrain = np.loadtxt('data/train_images.txt')
arrayTrainY = np.loadtxt('data/train_labels.txt')

#exercitiul 3

X_train = value_to_bins(arrayTrain, bins)

clf = MultinomialNB()
clf.fit(X_train, arrayTrainY)

X_test = value_to_bins(arrayTest, bins)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(arrayTestY, y_pred)
print(accuracy)

#exercitiul 4

num_bins_list = [3, 5, 7, 9, 11]
for num in num_bins_list:
    bins = np.linspace(0, 255, num)
    X1 = value_to_bins(arrayTrain, bins)
    X2 = value_to_bins(arrayTest, bins)

    clf = MultinomialNB()
    clf.fit(X1, arrayTrainY)

    y_pred = clf.predict(X2)

    accuracy = accuracy_score(arrayTestY, y_pred)
    print(accuracy)

#exercitiul 5

bins = np.linspace(0, 255, 11)
X1 = value_to_bins(arrayTrain, bins)
X2 = value_to_bins(arrayTest, bins)

clf = MultinomialNB()
clf.fit(X1, arrayTrainY)

y_pred = clf.predict(X2)
wrong_indices = np.where(y_pred != arrayTestY)[0]

for i in range(10):
    idx = wrong_indices[i]
    image = arrayTest[idx].reshape(28, 28)
    plt.imshow(image, cmap = 'gray')
    plt.title(f"True: {int(arrayTestY[idx])}, Predicted: {int(y_pred[idx])}")
    plt.show()


#exercitiul 6

matrix = confusion_matrix(arrayTestY, y_pred)
print(matrix)




