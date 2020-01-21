from data import get_training_data_4d, get_training_data_2d
import cnn, rf, svm, nn5, dt, nb


data = get_training_data_4d()

_cnn_model = cnn.load_model()
_cnn_cm = cnn.evaluate_cm(_cnn_model,data)
print(f"{cnn.name}\n{_cnn_cm}\n")

data = get_training_data_2d()

_nn5_model = nn5.load_model()
_nn5_cm = nn5.evaluate_cm(_nn5_model,data)
print(f"{nn5.name}\n{_nn5_cm*100}\n")

_svm_model = svm.load_model()
_svm_cm = svm.evaluate_cm(_svm_model,data)
print(f"{svm.name}\n{_svm_cm*100}\n")

_rf_model = rf.load_model()
_rf_cm = rf.evaluate_cm(_rf_model,data)
print(f"{rf.name}\n{_rf_cm}\n")

_dt_model = dt.load_model()
_dt_cm = dt.evaluate_cm(_dt_model,data)
print(f"{dt.name}\n{_dt_cm*100}\n")

_nb_model = nb.load_model()
_nb_cm = nb.evaluate_cm(_nb_model,data)
print(f"{nb.name}\n{_nb_cm*100}\n")