from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

svm_linear = SVC(kernel="linear", C=1.0)
svm_linear.fit(X_train, y_train)

y_predict = svm_linear.predict(X_test)

print("The accuracy of the svm model(liner) is " + str(accuracy_score(y_test,y_predict)*100) + " %")

print("\nClassification Report:\n", classification_report(y_test, y_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_predict))