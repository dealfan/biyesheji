import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression  # 新增元模型
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib
from tqdm import tqdm

# 加载预处理数据
TRAIN_PROCESSED = 'processed/train_processed.csv'
VAL_PROCESSED = 'processed/val_processed.csv'
TFIDF_PICKLE = 'processed/tfidf_vectorizer.pkl'

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    return df

def load_vectorizer(pickle_path):
    with open(pickle_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

def tune_model(model, param_grid, X_train, y_train, model_name):
    """
    使用 GridSearchCV 对模型进行参数调优，返回最佳模型。
    """
    print(f"\n正在对 {model_name} 进行参数网格搜索...")
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"{model_name} 最优参数: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_models(X_train, y_train):
    # 学习器参数调优
    # MultinomialNB 参数网格
    nb_param_grid = {'alpha': [0.5, 1.0, 1.5]}
    nb_clf = tune_model(MultinomialNB(), nb_param_grid, X_train, y_train, 'MultinomialNB')

    # SVC 参数网格（线性核，概率型）
    svc_param_grid = {'C': [0.5, 1, 2], 'probability': [True], 'kernel': ['linear']}
    svc_clf = tune_model(SVC(random_state=42), svc_param_grid, X_train, y_train, 'SVC')

    # RandomForest 参数网格
    rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'random_state': [42]}
    rf_clf = tune_model(RandomForestClassifier(), rf_param_grid, X_train, y_train, 'RandomForest')

    models = [
        ("MultinomialNB", nb_clf),
        # ("SVC", svc_clf),
        ("RandomForest", rf_clf)
    ]
    
    # 构造 Stacking 集成模型（逻辑回归作为元模型）
    stacking_clf = StackingClassifier(
        estimators=models,
        final_estimator=LogisticRegression(max_iter=1000),  # 元模型
        cv=5  # 交叉验证折数
    )
    
    print("\n开始训练 Stacking 集成模型...")
    stacking_clf.fit(X_train, y_train)
    print("Stacking 模型训练完成")
    
    return stacking_clf

# 加载数据
train_df = load_data(TRAIN_PROCESSED)
vectorizer = load_vectorizer(TFIDF_PICKLE)

# 由TF-IDF向量器转换文本特征
X_train = vectorizer.transform(train_df['text_processed'])
y_train = train_df['label']

# 模型训练
ensemble_model = train_models(X_train, y_train)

# 评估在训练集上的简单分类报告（后续调试过程中可加入验证集调参）
preds = ensemble_model.predict(X_train)
print("训练集评估报告：")
print(classification_report(y_train, preds))

# 保存模型，后续在线调用时加载
joblib.dump(ensemble_model, 'stacking_news_model.pkl')
print("模型训练完成")