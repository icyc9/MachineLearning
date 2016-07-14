from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()


classifier = svm.SVC(gamma=0.001, C=100.)
# Fit all but the last digit
classifier.fit(digits.data[:-1], digits.target[:-1])
# Predict the last digit in data
prediction = classifier.predict(digits.data[-1:])

print("Predicted %s" % prediction[0])