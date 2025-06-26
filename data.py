import pandas as pd
import jieba
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

TRAIN_FILE = 'data/cnews.train.txt'
VAL_FILE = 'data/cnews.val.txt'
TEST_FILE = 'data/cnews.test.txt'
STOPWORDS_FILE = 'data/cnews.vocab.txt'

def load_stopwords(stopwords_file):
    """加载停用词表"""
    stopwords = set()
    if os.path.exists(stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    return stopwords

def clean_text(text):
    # 去除数字和特殊符号
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    return text

def tokenize_text(text, stopwords):
    # 利用 jieba 分词
    text = clean_text(text)
    tokens = jieba.lcut(text)
    # 剔除停用词
    tokens = [tok for tok in tokens if tok.strip() and tok not in stopwords]
    return tokens

def load_dataset(file_path, stopwords):

    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
    # 将文本转换为字符串（str(x)）
    # 调用tokenize_text函数进行分词并过滤停用词
    # 将分词结果用空格连接（TF-IDF向量化需要以空格分隔的词语）
    df['text_processed'] = df['text'].apply(lambda x: " ".join(tokenize_text(str(x), stopwords)))
    return df

# max_features最大特征数，默认为5000
def build_tfidf_vectorizer(train_texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    return vectorizer

stopwords = load_stopwords(STOPWORDS_FILE)
print("加载停用词数量：", len(stopwords))

# 加载各数据集
train_df = load_dataset(TRAIN_FILE, stopwords)
val_df = load_dataset(VAL_FILE, stopwords)
test_df = load_dataset(TEST_FILE, stopwords)

print("训练集样本数：", train_df.shape[0])
print("验证集样本数：", val_df.shape[0])
print("测试集样本数：", test_df.shape[0])

# 构建 TF-IDF 向量器，并对文本进行向量转换
vectorizer = build_tfidf_vectorizer(train_df['text_processed'])
X_train = vectorizer.transform(train_df['text_processed'])
X_val = vectorizer.transform(val_df['text_processed'])
X_test = vectorizer.transform(test_df['text_processed'])

# 保存向量器供后续使用
with open('processed/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# 保存预处理后的数据（也可以直接在后续代码中调用 DataFrame）
train_df.to_csv('processed/train_processed.csv', index=False, encoding='utf-8')
val_df.to_csv('processed/val_processed.csv', index=False, encoding='utf-8')
test_df.to_csv('processed/test_processed.csv', index=False, encoding='utf-8')

print("数据预处理完成，并保存向量器与预处理后的数据。")