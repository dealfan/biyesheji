import pandas as pd
import pickle
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

TEST_PROCESSED = 'processed/test_processed.csv'
TFIDF_PICKLE = 'processed/tfidf_vectorizer.pkl'
MODEL_PICKLE = 'stacking_news_model.pkl'

def load_data(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

def load_vectorizer(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def plot_label_distribution(y_true):
    """绘制测试集真实标签的类别分布"""
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=y_true, order=y_true.value_counts().index)
    ax.set_title("测试集真实标签分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_classification_metrics(report_dict):
    """绘制每类的Precision、Recall和F1柱状图"""
    df = pd.DataFrame(report_dict).transpose()
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'])  # 只保留具体类别

    df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
    plt.title('各类别指标对比')
    plt.xlabel('类别')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
# 加载数据和模型
test_df = load_data(TEST_PROCESSED)
vectorizer = load_vectorizer(TFIDF_PICKLE)
model = joblib.load(MODEL_PICKLE)
if isinstance(model, tuple):  # 处理可能返回元组的情况
    model = model[0]

X_test = vectorizer.transform(test_df['text_processed'])
y_test = test_df['label']
y_pred = model.predict(X_test)

# 输出分类结果
print("✅ 测试集准确率：", accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred, output_dict=True)
print("✅ 分类报告：")
print(classification_report(y_test, y_pred))

labels = sorted(test_df['label'].unique())

# ✅可视化内容
plot_label_distribution(y_test)
plot_confusion_matrix(y_test, y_pred, labels)
plot_classification_metrics(report)