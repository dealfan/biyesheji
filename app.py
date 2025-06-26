import streamlit as st
import pandas as pd
import joblib
import pickle
import jieba
import re
import io

# 自定义 CSS 样式
st.markdown(
    """
    <style>
        body {
            background-color: #f9f9f9;
        }
        .stSidebar {
            background: #fff6f0;
        }
        .stRadio input[type=\"radio\"] {
            accent-color: #FF5733;
        }
        .stRadio label {
            font-size: 16px;
            color: #333;
        }
        .stRadio input[type=\"radio\"]:checked + label {
            color: #FF5733;
        }
        .stTextInput textarea {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
        }
        .stFileUploader div > div > input {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
        }
        .stDataFrame table {
            font-size: 14px;
            border-collapse: collapse;
            width: 100%;
        }
        .stDataFrame th, .stDataFrame td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .stDataFrame th {
            background-color: #f2f2f2;
        }
        .card {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            padding: 24px;
            margin-bottom: 24px;
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #FF5733;
            margin-bottom: 0.5em;
        }
        .sub-title {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 1.5em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 分类颜色映射
category_colors = {
    "体育": "#4CAF50",
    "财经": "#2196F3",
    "房产": "#9C27B0",
    "家居": "#FF9800",
    "教育": "#607D8B",
    "科技": "#795548",
    "时尚": "#E91E63",
    "时政": "#F44336",
    "游戏": "#00BCD4",
    "娱乐": "#8BC34A"
}

TFIDF_PICKLE = 'tfidf_vectorizer.pkl'
MODEL_PICKLE = 'ensemble_news_model.pkl'
STOPWORDS_FILE = 'cnews.vocab.txt'

@st.cache_resource
def load_vectorizer():
    with open(TFIDF_PICKLE, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PICKLE)

@st.cache_data
def load_stopwords():
    stopwords = set()
    try:
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    except Exception as e:
        st.error("停用词加载错误: " + str(e))
    return stopwords

def clean_text(text):
    text = re.sub(r'[^一-龥]', ' ', str(text))
    return text.strip()

def tokenize_text(text, stopwords):
    text = clean_text(text)
    if not text:
        return ""
    tokens = jieba.lcut(text)
    tokens = [tok for tok in tokens if tok.strip() and tok not in stopwords]
    return " ".join(tokens)

def classify_texts(texts, stopwords, vectorizer, model):
    processed = []
    non_empty_indices = []
    for i, text in enumerate(texts):
        cleaned_text = clean_text(text)
        if cleaned_text:
            processed.append(tokenize_text(cleaned_text, stopwords))
            non_empty_indices.append(i)
    if not processed:
        return []
    X_input = vectorizer.transform(processed)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)
        preds = model.predict(X_input)
        max_proba = proba.max(axis=1)
    else:
        preds = model.predict(X_input)
        max_proba = [None] * len(preds)
    full_preds = [""] * len(texts)
    full_probas = [None] * len(texts)
    for idx, pred, prob in zip(non_empty_indices, preds, max_proba):
        full_preds[idx] = pred
        full_probas[idx] = prob
    return list(zip(full_preds, full_probas))

# --- 页面定义 ---
def page_home():
    st.markdown('<div class="main-title">✨ 中文新闻分类系统 ✨</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">基于机器学习的多类别新闻文本智能分类平台</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>功能简介：</b><br>
    - 支持单条/多条文本输入分类<br>
    - 支持 TXT/CSV 文件批量上传分类<br>
    - 分类结果可下载<br>
    - 现代化美观界面，交互友好<br>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/000000/news.png", width=96)
    st.markdown("""
    <div style='margin-top:2em;'>
    <b>请通过左侧菜单选择功能页面进行体验。</b>
    </div>
    """, unsafe_allow_html=True)

def page_data_processing():
    st.markdown('<div class="main-title">🛠️ 数据处理</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">文本清理与分词工具，体验文本预处理效果</div>', unsafe_allow_html=True)
    stopwords = load_stopwords()
    text_input = st.text_area(
        "请输入待处理文本：",
        placeholder="",
        height=150,
        key="data_proc_textarea"
    )
    if st.button("文本清理", key="clean_btn"):
        if text_input.strip():
            cleaned_lines = [clean_text(line) for line in text_input.splitlines()]
            st.success("清理结果：")
            st.code("\n".join(cleaned_lines), language=None)
        else:
            st.warning("请输入有效的文本内容！")
    if st.button("分词处理", key="tokenize_btn"):
        if text_input.strip():
            tokenized_lines = [tokenize_text(line, stopwords) for line in text_input.splitlines()]
            st.success("分词结果：")
            st.code("\n".join(tokenized_lines), language=None)
        else:
            st.warning("请输入有效的文本内容！")

def page_text_classification():
    st.markdown('<div class="main-title">📝 文本分类</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">输入新闻文本，系统将自动为您分类</div>', unsafe_allow_html=True)
    stopwords = load_stopwords()
    vectorizer = load_vectorizer()
    model = load_model()
    text_input = st.text_area(
        "请输入新闻文本：",
        placeholder="",
        height=150
    )
    # 新增：合并并复制按钮功能
    merged_text = ""
    if text_input.strip():
        merged_text = " ".join(text_input.splitlines())
    if st.button("合并文本", key="merge_copy_btn"):
        if merged_text:
            st.code(merged_text, language=None)
            # st.success("已生成合并文本，可复制！")
        else:
            st.warning("请输入有效的文本内容！")
    if st.button("开始分类", key="text_classify_btn"):
        if text_input.strip():
            text = " ".join(text_input.splitlines())
            results = classify_texts([text], stopwords, vectorizer, model)
            prediction = results[0][0]
            confidence = results[0][1]
            result_df = pd.DataFrame({
                "输入文本": [text],
                "分类结果": [prediction],
                "置信度": [confidence]
            })
            st.success("🎉 分类完成！以下是结果：")
            
            # 添加分类结果卡片
            category_colors = {
                "体育": "#4CAF50",
                "财经": "#2196F3",
                "房产": "#9C27B0",
                "家居": "#FF9800",
                "教育": "#607D8B",
                "科技": "#795548",
                "时尚": "#E91E63",
                "时政": "#F44336",
                "游戏": "#00BCD4",
                "娱乐": "#8BC34A"
            }
            
            color = category_colors.get(prediction, "#607D8B")
            
            st.markdown(
                f"""
                <div class="card" style="border-left: 6px solid {color};">
                    <h3 style="color: {color};">{prediction}</h3>
                    <p><b>输入文本：</b> {text[:100]}...</p>
                    <p><b>置信度：</b> {confidence if confidence else 'N/A'}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 添加表格样式
            st.markdown(
                """
                <style>
                    .stDataFrame {
                        border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }
                    .stDataFrame th {
                        background-color: #FF5733 !important;
                        color: white !important;
                    }
                    .stDataFrame tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        else:
            st.warning("请输入有效的文本内容！")

def page_file_upload():
    st.markdown('<div class="main-title">📁 文件上传分类</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">上传 TXT 或 CSV 文件，批量进行新闻分类</div>', unsafe_allow_html=True)
    stopwords = load_stopwords()
    vectorizer = load_vectorizer()
    model = load_model()
    file = st.file_uploader(
        "选择文件上传",
        type=['txt', 'csv'],
        accept_multiple_files=False
    )
    if file is not None:
        try:
            df = pd.DataFrame()
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                st.write("检测到 CSV 文件")
                if df.empty:
                    st.error("CSV 文件为空，请上传有效文件！")
                    return
                text_col = st.selectbox("请选择文本列：", df.columns)
                texts = df[text_col].fillna("").tolist()
            else:
                st.write("检测到 TXT 文件，每行为一条新闻")
                texts = file.read().decode('utf-8').splitlines()
                df = pd.DataFrame({'text': texts})
            if df.empty or not texts:
                st.error("文件内容为空，请上传有效文件！")
                return
            predictions = classify_texts(texts, stopwords, vectorizer, model)
            df['分类结果'] = [r[0] for r in predictions]
            df['置信度'] = [r[1] for r in predictions]
            df.insert(0, '', range(1, len(df) + 1))
            st.success("🎉 分类完成！以下是部分结果：")
            
            # 添加表格样式
            st.markdown(
                """
                <style>
                    .stDataFrame {
                        border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }
                    .stDataFrame th {
                        background-color: #FF5733 !important;
                        color: white !important;
                    }
                    .stDataFrame tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # 注释掉分类结果卡片展示
            # sample_df = df.head(3)
            # for _, row in sample_df.iterrows():
            #     color = category_colors.get(row['分类结果'], "#607D8B")
            #     st.markdown(
            #         f"""
            #         <div class="card" style="border-left: 6px solid {color}; margin-bottom: 16px;">
            #             <h3 style="color: {color};">{row['分类结果']}</h3>
            #             <p><b>文本：</b> {row['text'][:100] if 'text' in row else row.iloc[0][:100]}...</p>
            #             <p><b>置信度：</b> {row['置信度'] if '置信度' in row else 'N/A'}</p>
            #         </div>
            #         """,
            #         unsafe_allow_html=True
            #     )
            
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            output = io.StringIO()
            df.to_csv(output, index=False, encoding='utf-8')
            st.download_button(
                label="点击下载分类结果CSV",
                data=output.getvalue(),
                file_name='classified_news.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"处理文件出错：{e}")

def page_about():
    st.markdown('<div class="main-title">ℹ️ 关于项目</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">项目介绍与开发者信息</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>项目名称：</b> 中文新闻智能分类系统<br>
    <b>主要功能：</b> 基于集成学习的新闻文本自动分类<br>
    <b>技术栈：</b> Streamlit, scikit-learn, jieba, pandas 等<br>
    <b>开发者：</b> 1193210421-范安明<br>
    <b>联系方式：</b> 2675433494@qq.com<br>
    <b>开源协议：</b> MIT License
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:2em;'>
    <b>感谢您的使用与支持！</b>
    </div>
    """, unsafe_allow_html=True)

# --- 登录页面 ---


# --- 主程序入口 ---
def main():

    st.sidebar.image("https://img.icons8.com/color/48/000000/news.png", width=48)
    st.sidebar.title("导航菜单")

    # 使用 session_state 记录当前页面
    if "sidebar_page" not in st.session_state:
        st.session_state["sidebar_page"] = "首页"
    
    pages = [
        ("首页", "🏠 首页"),
        ("文本分类", "📝 文本分类"),
        ("文件上传", "📁 文件上传"),
        ("数据处理", "🛠️ 数据处理"),
        ("关于项目", "ℹ️ 关于项目")
    ]
    
    for key, label in pages:
        btn = st.sidebar.button(label, key=f"sidebar_btn_{key}")
        if btn:
            st.session_state["sidebar_page"] = key
    
    page = st.session_state["sidebar_page"]
    
    # 自定义按钮样式
    st.markdown(
        """
        <style>
        .stSidebar button {
            background: #fff;
            color: #FF5733;
            border: 2px solid #FF5733;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 10px;
            transition: background 0.2s, color 0.2s;
            font-weight: 500;
        }
        .stSidebar button:hover {
            background: #FF5733;
            color: #fff;
        }
        .stSidebar button:focus {
            outline: none;
            box-shadow: 0 0 0 2px #FF573333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if page == "首页":
        page_home()
    elif page == "文本分类":
        page_text_classification()
    elif page == "文件上传":
        page_file_upload()
    elif page == "数据处理":
        page_data_processing()
    elif page == "关于项目":
        page_about()

if __name__ == '__main__':
    main()