import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score

def load_data(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
    header = data[0]
    data = np.array(data[1:])

    # Convert class labels to numerical values
    labels = {'positive': 1, 'negative': 0}
    for i in range(len(data)):
        data[i, -1] = labels[data[i, -1]]

    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(int)
    return X, y


X, y = load_data("Heart Attack.csv")


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVM()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

precision_score = precision_score(y_test, predictions)
recall_score = recall_score(y_test, predictions)

print("Precision Score:", precision_score)
print("Recall Score:", recall_score)