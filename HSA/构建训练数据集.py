import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取完整数据
data_path = 'data/train/training_dataset_final.csv'
df = pd.read_csv(data_path)

# 输入特征列 (7个)
feature_cols = [
    'A_in_10^3m2', 'A_out_10^3m2',
    'E_in_MWh', 'E_out_MWh',
    'P_Tr_max_in_MW', 'P_Tr_max_out_MW',
    'P_mut_max_MW'
]

# ==========================================
# 数据集 A: 用于训练分类器 (Classifier)
# ==========================================
X_cls = df[feature_cols].values
y_cls = df['feasible'].values  # 0 或 1

# 划分训练/测试集
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

print(f"分类器数据准备就绪: 训练样本 {len(X_train_cls)} 个")

# ==========================================
# 数据集 B: 用于训练预测器 (Predictor)
# ==========================================
# 关键步骤: 只筛选出"可行"的样本
df_feasible = df[df['feasible'] == 1].copy()

X_reg = df_feasible[feature_cols].values
y_reg = df_feasible['f2_operation'].values # 目标是 f2

# 归一化 (对于回归网络，Target最好也做归一化，或者取对数)
# y_reg = np.log1p(y_reg) # 可选技巧

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"预测器数据准备就绪: 训练样本 {len(X_train_reg)} 个 (已剔除不可行样本)")

# ==========================================
# 至于 f1...
# ==========================================
print("提示: f1 列的数据不需要用于训练，它只在后续 NSGA-II 中作为公式验证对比使用。")