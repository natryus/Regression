import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv('dataset_v2.csv')

X = data.drop(['timestamp', 'bitrate'], axis=1)

y = data['bitrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressorMLP = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=400, random_state=42)
regressor_rf = RandomForestRegressor(max_depth=3, n_estimators=8, random_state=42)
regressor_knn = KNeighborsRegressor(n_neighbors=3, algorithm='auto', leaf_size=10,
                                      metric='euclidean', metric_params=None, n_jobs=1, p=2, weights='uniform')
regressor_svm = SVR(degree=3, kernel='rbf', C=1.0, gamma='auto',epsilon=0.1)


regressorMLP.fit(X_train.values, y_train.values)

score_mlp = regressorMLP.score(X_test.values, y_test.values)

filename = 'Models/regressor_mlp.joblib'

joblib.dump(regressorMLP, filename)
regressor_mlp_loaded = joblib.load(filename)

y_pred_mlp = regressor_mlp_loaded.predict(X_test.values).ravel()
mse_mlp = mean_squared_error(y_test.values, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print(y_pred_mlp)
print("Erro médio quadrático do Regressor com MLP: {:.4f}".format(mse_mlp))
print("Raiz do erro médio quadrático do Regressor com MLP: {:.4f}".format(rmse_mlp))
print("R2 Score com MLP: {:.4f}".format(r2_mlp))

plt.figure(num=1, figsize=(8, 6))
plt.scatter(y_test, y_pred_mlp, marker='o', label='Observation')
plt.plot(y_test, y_test, color='black', linestyle='--', linewidth=2, label='Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values - ANN')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()


regressor_rf.fit(X_train, y_train)
score_rf = regressor_rf.score(X_test.values, y_test.values)

filename = 'Models/regressor_rf.joblib'

joblib.dump(regressor_rf, filename)
regressor_rf_loaded = joblib.load(filename)

y_pred_rf = regressor_rf_loaded.predict(X_test.values).ravel()
mse_rf = mean_squared_error(y_test.values, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(y_pred_rf)
print("Erro médio quadrático do Regressor com RF: {:.4f}".format(mse_rf))
print("Raiz do erro médio quadrático do Regressor com RF: {:.4f}".format(rmse_rf))
print("R2 Score com RF: {:.4f}".format(r2_rf))

plt.figure(num=2, figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, marker='o', label='Observation')
plt.plot(y_test, y_test, color='black', linestyle='--', linewidth=2, label='Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values - RF')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()


regressor_knn.fit(X_train, y_train)
score_knn = regressor_knn.score(X_test.values, y_test.values)

filename = 'Models/regressor_knn.joblib'

joblib.dump(regressor_knn, filename)
regressor_knn_loaded = joblib.load(filename)
print(regressor_knn_loaded)
y_pred_knn = regressor_knn_loaded.predict(X_test.values).ravel()
mse_knn = mean_squared_error(y_test.values, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(y_pred_knn)
print("Erro médio quadrático do Regressor com KNN: {:.4f}".format(mse_knn))
print("Raiz do erro médio quadrático do Regressor com KNN: {:.4f}".format(rmse_knn))
print("R2 Score com KNN: {:.4f}".format(r2_knn))

plt.figure(num=3, figsize=(8, 6))
plt.scatter(y_test, y_pred_knn, marker='o', label='Observation')
plt.plot(y_test, y_test, color='black', linestyle='--', linewidth=2, label='Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values - KNN')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()


regressor_svm.fit(X_train, y_train)
score_svm = regressor_svm.score(X_test.values, y_test.values)

filename = 'Models/regressor_svm.joblib'

joblib.dump(regressor_svm, filename)
regressor_svm_loaded = joblib.load(filename)

y_pred_svm = regressor_svm_loaded.predict(X_test.values).ravel()
mse_svm = mean_squared_error(y_test.values, y_pred_svm)
rmse_svm = np.sqrt(mse_svm)
r2_svm = r2_score(y_test, y_pred_svm)

print(y_pred_svm)
print("Erro médio quadrático do Regressor com SVM: {:.4f}".format(mse_svm))
print("Raiz do erro médio quadrático do Regressor com SVM: {:.4f}".format(rmse_svm))
print("R2 Score com SVM: {:.4f}".format(r2_svm))

plt.figure(num=4, figsize=(8, 6))
plt.scatter(y_test, y_pred_svm, marker='o', label='Observation')
plt.plot(y_test, y_test, color='black', linestyle='--', linewidth=2, label='Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values - SVM')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.show()
