from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.metrics import AUC, Precision
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr

results = []

def train_and_evaluate(num_point):
    # 更新输入特征的数量和设施数量
    input_shape = (num_point ** 2 + 2 * num_point + 1,)
    num_facilities = num_point * num_point

    # 按照命名规则加载数据
    X_train = np.load(f'X_train_b{num_point}.npy')
    Y_train_class = np.load(f'Y_train_class_b{num_point}.npy')
    Y_train_regress = np.load(f'Y_train_regress_b{num_point}.npy')
    X_test = np.load(f'X_test_b{num_point}.npy')
    Y_test_class = np.load(f'Y_test_class_b{num_point}.npy')
    Y_test_regress = np.load(f'Y_test_regress_b{num_point}.npy')

    # 分类模型定义
    model_classification = Sequential([
        Dense(256, activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_facilities, activation='sigmoid')
    ])
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
        Dense(1, activation='linear')
    ])
    model_regression.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # 早停策略和学习率调度器
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # 拟合分类模型
    model_classification.fit(X_train, Y_train_class, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping_monitor, reduce_lr])

    # 拟合回归模型
    model_regression.fit(X_train, Y_train_regress, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping_monitor, reduce_lr])

    # 评估回归模型
    loss_regress, mae_regress = model_regression.evaluate(X_test, Y_test_regress)
    y_pred_test_regress = model_regression.predict(X_test).flatten()
    pearson_corr_test, _ = pearsonr(Y_test_regress.flatten(), y_pred_test_regress)

    # 评估分类模型
    loss_class, accuracy_class, auc_class, precision_class = model_classification.evaluate(X_test, Y_test_class)
    y_pred_test_class = model_classification.predict(X_test)

    # 检查测试标签的多样性
    if len(np.unique(Y_test_class)) > 1:
        auc_score = roc_auc_score(Y_test_class, y_pred_test_class)
        auprc_score = average_precision_score(Y_test_class, y_pred_test_class)
    else:
        auc_score = None
        auprc_score = None

    # 返回结果
    return {
        'num_point': num_point,
        'mae_regress': mae_regress,
        'pearson_corr_test': pearson_corr_test,
        'auc_score': auc_score,
        'auprc_score': auprc_score
    }

# 主循环
for num_point in range(3, 11):
    result = train_and_evaluate(num_point)
    results.append(result)
    print(f"num_point: {num_point}, Results: {result}")

# 输出结果
for result in results:
    print(result)
