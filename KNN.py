from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNN:

    def __init__(self, k=5):
        self.model = KNeighborsClassifier(n_neighbors=k)

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        y_pred = self.model.predict(x)
        return accuracy_score(y, y_pred)