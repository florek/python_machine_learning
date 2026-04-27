from sklearn.datasets import load_iris


def load_iris_data():
    X_full, y_full = load_iris(return_X_y=True)
    X = X_full[0:100, [0, 2]]
    y = y_full[0:100]
    return X, y
