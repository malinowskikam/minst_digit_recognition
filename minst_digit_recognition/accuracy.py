from data import get_training_data_4d, get_training_data_2d
import cnn, rf, svm, nn5, dt, nb


data = get_training_data_4d()

_cnn_model = cnn.load_model()
_cnn_acc = cnn.evaluate_acc(_cnn_model,data)
print(f"{cnn.name} - {_cnn_acc*100}%")

data = get_training_data_2d()

_nn5_model = nn5.load_model()
_nn5_acc = nn5.evaluate_acc(_nn5_model,data)
print(f"{nn5.name} - {_nn5_acc*100}%")

_svm_model = svm.load_model()
_svm_acc = svm.evaluate_acc(_svm_model,data)
print(f"{svm.name} - {_svm_acc*100}%")

_rf_model = rf.load_model()
_rf_acc = rf.evaluate_acc(_rf_model,data)
print(f"{rf.name} - {_rf_acc*100}%")

_dt_model = dt.load_model()
_dt_acc = dt.evaluate_acc(_dt_model,data)
print(f"{dt.name} - {_dt_acc*100}%")

_nb_model = nb.load_model()
_nb_acc = nb.evaluate_acc(_nb_model,data)
print(f"{nb.name} - {_nb_acc*100}%")
