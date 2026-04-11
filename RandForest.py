from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandForest:

    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        y_pred = self.model.predict(x)
        return accuracy_score(y, y_pred)