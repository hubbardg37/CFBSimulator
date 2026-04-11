from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LogRegress:
    def __init__(self):
        self.model = LogisticRegression(max_iter=5000)

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        y_pred = self.model.predict(x)
        return accuracy_score(y, y_pred)