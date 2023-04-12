import joblib


def load_model(name):
    model = joblib.load('model.joblib')
    return


def save_model(model, name):
    joblib.dump(model, 'model.joblib')




