import numpy as np
import matplotlib.pyplot as plt


#execitiul 1 & 2

test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt')
train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt')

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def predict(self, test_image, k=3):

        distances = np.linalg.norm(self.train_images - test_image, axis=1)

        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = self.train_labels[nearest_indices]

        values, counts = np.unique(nearest_labels, return_counts=True)
        return values[np.argmax(counts)]

    def classify_image(self, test_image, num_neighbours=3, metric='l2'):
        if metric == 'l1':
            distance = np.sum(np.abs(self.train_images - test_image), axis=1)
        elif metric == 'l2':
            distance = np.linalg.norm(self.train_images - test_image, axis=1)

        nearest_indices = np.argsort(distance)[:num_neighbours]
        nearest_labels = self.train_labels[nearest_indices]

        values, counts = np.unique(nearest_labels, return_counts=True)
        return values[np.argmax(counts)]


#exercitiul 3

knn = KnnClassifier(train_images, train_labels)

predictions = []
for image in test_images:
    predictions.append(knn.classify_image(image))
predictions = np.array(predictions)

accuracy = np.mean(test_labels == predictions)

print(accuracy)

np.savetxt('predictii_3nn_l2_mnist.txt', predictions, fmt='%d')

#exercitiul 4

score = []
neighbors = [1, 3, 5, 7, 9]
for num in neighbors:
    predictions = []
    for image in test_images:
        predictions.append(knn.classify_image(image, num, 'l2'))
    predictions = np.array(predictions)
    accuracy = np.mean(test_labels == predictions)
    print(accuracy)
    score.append(accuracy)

np.savetxt('acuratete_I2.txt', score, fmt='%.3f')

plt.plot(neighbors, score)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

score2 = []
neighbors = [1, 3, 5, 7, 9]
for num in neighbors:
    predictions = []
    for image in test_images:
        predictions.append(knn.classify_image(image, num, 'l1'))
    predictions = np.array(predictions)
    accuracy = np.mean(test_labels == predictions)
    print(accuracy)
    score2.append(accuracy)

plt.plot(neighbors, score)
plt.plot(neighbors, score2)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()