import matplotlib.pyplot as plt

# 数据
data = [
    {'num_point': 3, 'mae_regress': 5.416170597076416, 'pearson_corr_test': 0.9832177582290567, 'auc_score': 0.987664480048746},
    {'num_point': 4, 'mae_regress': 9.329793930053711, 'pearson_corr_test': 0.9734696620634679, 'auc_score': 0.502107640693237},
    {'num_point': 5, 'mae_regress': 10.580743789672852, 'pearson_corr_test': 0.9720166511599905, 'auc_score': 0.7604462826540208},
    {'num_point': 6, 'mae_regress': 13.31871223449707, 'pearson_corr_test': 0.9687285629591624, 'auc_score': 0.7101235399289453},
    {'num_point': 7, 'mae_regress': 15.466015815734863, 'pearson_corr_test': 0.9671204382401939, 'auc_score': 0.4986218402757824},
    {'num_point': 8, 'mae_regress': 18.948366165161133, 'pearson_corr_test': 0.9657378949153361, 'auc_score': 0.598745376908313},
    {'num_point': 9, 'mae_regress': 20.375946044921875, 'pearson_corr_test': 0.9635901162571373, 'auc_score': 0.513566050178049},
    {'num_point': 10, 'mae_regress': 21.5395565032959, 'pearson_corr_test': 0.9670234050514406, 'auc_score': 0.4976418085168683}
]

# 提取数据
num_points = [d['num_point'] for d in data]
pearson_corrs = [d['pearson_corr_test'] for d in data]
auc_scores = [d['auc_score'] for d in data]

# 重新绘图，并确保坐标轴始终显示0-1的范围

plt.figure(figsize=(10, 5))

# Pearson correlation
plt.subplot(1, 2, 1)
plt.plot(num_points, pearson_corrs, marker='o')
plt.title('Pearson Correlation')
plt.xlabel('Number of Points')
plt.ylabel('Pearson Correlation')
plt.ylim(0, 1)  # 设置y轴范围为0-1
plt.gca().lines[0].set_clip_on(False)  # 确保首位数据不链接

# AUC
plt.subplot(1, 2, 2)
plt.plot(num_points, auc_scores, marker='o', color='orange')
plt.title('AUC Score')
plt.xlabel('Number of Points')
plt.ylabel('AUC Score')
plt.ylim(0, 1)  # 设置y轴范围为0-1
plt.gca().lines[0].set_clip_on(False)  # 确保首位数据不链接

plt.tight_layout()
plt.show()
