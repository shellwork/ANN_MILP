from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.metrics import AUC, Precision
import numpy as np
from data_process import training_data
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr

num_point = 4

input_shape = (num_point ** 2 + 2 * num_point + 1,)
num_facilities = num_point * num_point  # 每个设施一个输出

# 输入特征的数量
input_shape = (num_point ** 2 + 2 * num_point + 1,)
X_train = np.load('X_train.npy')
Y_train_class = np.load('Y_train_class.npy')
Y_train_regress = np.load('Y_train_regress.npy')
X_test = np.load('X_test.npy')
Y_test_class = np.load('Y_test_class.npy')
Y_test_regress = np.load('Y_test_regress.npy')

# 确认训练数据形状
print(f"X_train shape: {X_train.shape}")
print(f"Y_train_class shape: {Y_train_class.shape}")
print(f"Y_train_regress shape: {Y_train_regress.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test_class shape: {Y_test_class.shape}")
print(f"Y_test_regress shape: {Y_test_regress.shape}")

# 输入特征的数量
input_shape = (num_point ** 2 + 2 * num_point + 1,)

# 分类模型定义
model_classification = Sequential([
    Dense(256, activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_facilities, activation='sigmoid')  # 输出num_facilities个值，使用sigmoid激活函数
])

# 在模型定义中添加AUC和Precision作为评估指标
auc_metric = AUC(curve='ROC', name='auc')
precision_metric = Precision(name='precision')

model_classification.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=['accuracy', auc_metric, precision_metric])

# 回归模型定义
model_regression = Sequential([
    Dense(128, activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # 单一输出，预测成本
])

# 编译回归模型
model_regression.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 早停策略
early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# 学习率调度器
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# 拟合分类模型
model_classification.fit(X_train, Y_train_class, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping_monitor, reduce_lr])

# 拟合回归模型
model_regression.fit(X_train, Y_train_regress, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping_monitor, reduce_lr])

def check_variability(data, name="Data"):
    print(f"{name} - Min: {np.min(data)}, Max: {np.max(data)}, Std: {np.std(data)}")

# 评估回归模型
loss_regress, mae_regress = model_regression.evaluate(X_test, Y_test_regress)
print(f"Regression Test Loss: {loss_regress}, Test MAE: {mae_regress}")
y_pred_test_regress = model_regression.predict(X_test).flatten()
check_variability(Y_test_regress, "Regression Test Targets")
check_variability(y_pred_test_regress, "Regression Test Predictions")
pearson_corr_test, _ = pearsonr(Y_test_regress.flatten(), y_pred_test_regress)
print(f"Test Pearson Correlation for Regression Model: {pearson_corr_test:.5f}")

# 评估分类模型
loss_class, accuracy_class, auc_class, precision_class = model_classification.evaluate(X_test, Y_test_class)
print(f"Classification Test Loss: {loss_class}, Test Accuracy: {accuracy_class}, AUC: {auc_class}, Precision: {precision_class}")
y_pred_test_class = model_classification.predict(X_test)
auc_score = roc_auc_score(Y_test_class, y_pred_test_class)
auprc_score = average_precision_score(Y_test_class, y_pred_test_class)
print(f"AUC: {auc_score:.4f}")
print(f"AUPRC: {auprc_score:.4f}")
check_variability(Y_test_class, "Test Targets")
check_variability(y_pred_test_class, "Test Predictions")

# model_classification.save('classification_model.h5')
# model_regression.save('regression_model.h5')