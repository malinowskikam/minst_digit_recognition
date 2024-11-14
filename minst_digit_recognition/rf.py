from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model_dir = "models/"
name = "random_forest"


def create_model():
    return RandomForestClassifier(n_jobs=-1, n_estimators=20, random_state=234263)


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
