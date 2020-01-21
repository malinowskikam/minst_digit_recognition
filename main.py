import nb
from data import get_training_data_2d

model = nb.create_model()
data = get_training_data_2d()
nb.fit(model,data)
print(nb.evaluate_acc(model,data))