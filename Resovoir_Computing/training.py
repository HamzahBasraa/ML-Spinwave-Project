import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# X = reservoir features
# y = digit labels (0-9)

X = np.load("features.npy")
y = np.load("labels.npy")

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RidgeClassifier()

# train
model.fit(X_train, y_train)

# test
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)