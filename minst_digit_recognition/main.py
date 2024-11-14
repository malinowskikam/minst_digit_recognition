from minst_digit_recognition import cnn, rf, svm, nn5, dt, nb
from minst_digit_recognition.data import get_training_data_2d, get_training_data_4d

data = get_training_data_2d()

model = nb.create_model()
nb.fit(model, data)

model = dt.create_model()
dt.fit(model, data)

model = nn5.create_model()
nn5.fit(model, data)

model = svm.create_model()
svm.fit(model, data)

model = rf.create_model()
dt.fit(model, data)


data = get_training_data_4d()

model = cnn.create_model()
cnn.fit(model, data)