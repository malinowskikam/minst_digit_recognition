from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

model_dir = "models/"
name = "decision_tree"


def create_model():
    return DecisionTreeClassifier()


def load_model():
    return load(model_dir + name + '.joblib')


def fit(model,data):
    model.fit(data[0],data[2])
    dump(model, model_dir + name + '.joblib')


def evaluate_acc(model,data):
    return model.score(data[1],data[3])


def evaluate_cm(model,data):
    prediction = model.predict(data[1])
    return confusion_matrix(data[3],prediction, labels=[0,1,2,3,4,5,6,7,8,9,])