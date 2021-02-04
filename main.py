from imblearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



from scipy import stats
import numpy as np

def z_score_outlier_detection(data, y):
    z = np.abs(stats.zscore(data))
    return data[(z < 3).all(axis=1)]

def iqr_outlier_detection(data, y):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]


class ClassifierWithImbalanceClass:
    def __init__(self):
        # self._pipeline = make_pipeline(StandardScaler(), LinearSVC())
        self._pipeline = make_pipeline(SMOTE(random_state=42),
                                       PCA(random_state=42),
                                       StandardScaler(),
                                       LinearSVC(random_state=42))

    def train(self, x, y):
        self.classifier = self._pipeline
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)


if __name__ == '__main__':
    import pandas as pd

    from sklearn.metrics import roc_auc_score


    train_data = pd.read_csv("/home/albert/Descargas/train_data.csv")
    y_train = train_data["target"]
    X_train = train_data.drop("target", axis="columns")

    test_data = pd.read_csv("/home/albert/Descargas/test_data.csv")
    y_test = test_data["target"]
    X_test = test_data.drop("target", axis="columns")

    model = ClassifierWithImbalanceClass()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    score = roc_auc_score(y_test, y_pred)
    print("La score obtenida es de", score)
