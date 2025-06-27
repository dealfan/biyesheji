import streamlit as st
import pandas as pd
import joblib
import pickle
import jieba
import re
import io
import plotly.express as px
from database import init_db, add_user, verify_user, get_all_users, delete_user, add_record, get_user_records, get_all_records
import numpy as np

init_db()
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
        .stRadio input[type="radio"] {
            accent-color: #FF5733;
        }
        .stRadio label {
            font-size: 16px;
            color: #333;
        }
        .stRadio input[type="radio"]:checked + label {
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
MODEL_PICKLE = 'stacking_news_model.pkl'
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
    # 确保至少返回一个有效结果
    processed = []
    non_empty_indices = []
    for i, text in enumerate(texts):
        cleaned_text = clean_text(text)
        if cleaned_text:
            processed.append(tokenize_text(cleaned_text, stopwords))
            non_empty_indices.append(i)
    
    # 确保至少有一个处理后的文本
    if not processed:
        # 返回默认分类结果
        return [('体育', 0.5)]  # 默认类别和置信度
    
    X_input = vectorizer.transform(processed)
    try:
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input) if hasattr(model, 'predict_proba') else [None]*len(processed)
    except Exception as e:
        print(f"预测过程出错: {e}")
        return [('体育', 0.5)]  # 预测失败时返回默认结果
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        # 确保每个结果都是元组格式
        if isinstance(prob, np.ndarray):
            results.append( (pred, prob.max()) )
        else:
            results.append( (pred, 0.5) )  # 默认置信度
    
    # 还原原始输入顺序
    full_results = [('体育', 0.5)] * len(texts)  # 默认填充
    for idx, res in zip(non_empty_indices, results):
        full_results[idx] = res
    return full_results

#  页面定义
def page_home():
    st.markdown('<div class="main-title">✨ 中文新闻分类系统 ✨</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">基于机器学习的多类别新闻文本智能分类平台</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>功能简介：</b><br>
    - 支持单条/多条文本输入分类<br>
    - 支持 TXT/CSV 文件批量上传分类<br>
    - 分类结果可下载<br>
    - 界面美观，交互友好<br>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/000000/news.png ", width=96)
    st.markdown("""
    <div style='margin-top:2em;'>
    <b>请通过左侧菜单选择功能页面进行体验。</b>
    </div>
    """, unsafe_allow_html=True)

def page_text_classification():
    st.markdown('<div class="main-title">📝 文本分类</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">输入段新闻文本，系统将自动为您分类</div>', unsafe_allow_html=True)
    stopwords = load_stopwords()
    vectorizer = load_vectorizer()
    # 直接加载模型
    model = load_model()
    text_input = st.text_area(
        "请输入新闻文本：",
        placeholder="",
        height=150
    )
    if st.button("开始分类", key="text_classify_btn"):
        if text_input.strip():
            text = " ".join(text_input.splitlines())
            results = classify_texts([text], stopwords, vectorizer, model)
            # 检查结果是否有效
            if not results or len(results) == 0:
                st.error("无法获取预测结果，请输入有效的文本内容。")
                prediction = None
                confidence = None
            else:
                prediction = results[0][0]
                confidence = results[0][1] if len(results[0]) > 1 else None
            result_df = pd.DataFrame({
                "输入文本": [text],
                "分类结果": [prediction],
                "置信度": [confidence]
            })
            st.success("🎉 分类完成！以下是结果：")
            # 分类结果卡片
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
            # 表格样式
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
            
            # 保存记录
            if "history_records" not in st.session_state:
                st.session_state["history_records"] = []
            record = {
                "type": "文本分类",
                "input": text,
                "result": prediction,
                "confidence": confidence,
                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state["history_records"].append(record)
            add_record(st.session_state["user"]["id"], record)
        else:
            st.warning("请输入有效的文本内容！")

def page_file_upload():
    st.markdown('<div class="main-title">📁 文件上传分类</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">上传 TXT 或 CSV 文件，批量进行新闻分类</div>', unsafe_allow_html=True)
    stopwords = load_stopwords()
    vectorizer = load_vectorizer()
    # 直接加载模型
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
            # 表格样式
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
            
            # 柱状图展示分类结果分布
            if not df.empty:
                category_counts = df['分类结果'].value_counts()
                fig = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title='分类结果分布',
                    color=category_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    text=category_counts.values
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            output = io.StringIO()
            df.to_csv(output, index=False, encoding='utf-8')
            st.download_button(
                label="点击下载分类结果CSV",
                data=output.getvalue(),
                file_name='classified_news.csv',
                mime='text/csv'
            )
            
            # 保存记录
            if "history_records" not in st.session_state:
                st.session_state["history_records"] = []
            record = {
                "type": "文件分类",
                "filename": file.name,
                "total": len(df),
                "results": df.head(10).to_dict(orient="records"),
                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state["history_records"].append(record)
            add_record(st.session_state["user"]["id"], record)
            
        except Exception as e:
            st.error(f"处理文件出错：{e}")



def page_history():
    st.markdown('<div class="main-title">📜 历史记录</div>', unsafe_allow_html=True)
    user = st.session_state.get("user")
    if not user:
        st.info("请先登录以查看历史记录")
        return
    records = get_user_records(user["id"])
    if not records:
        st.info("暂无历史记录")
        return
    for i, record in enumerate(records, 1):
        with st.expander(f"记录 #{i} - {record['time']}"):
            if record["type"] == "文本分类":
                st.markdown(f"**类型**: {record['type']}")
                st.markdown(f"**输入文本**: {record['input'][:100]}...")
                color = category_colors.get(record['result'], "#607D8B")
                st.markdown(f"<span style='color:{color}'>**分类结果**: {record['result']}</span>", unsafe_allow_html=True)
                st.markdown(f"**置信度**: {record['confidence']:.2%}" if record['confidence'] else "**置信度**: N/A")
            elif record["type"] == "文件分类":
                st.markdown(f"**类型**: {record['type']}")
                st.markdown(f"**文件名**: {record['filename']}")
                st.markdown(f"**总条数**: {record['total']}")
                st.markdown("**分类结果预览**: ")
                st.dataframe(pd.DataFrame(eval(record['results'])), hide_index=True)
    
    # 历史记录与数据库同步
    if "history_records" not in st.session_state:
        st.session_state["history_records"] = records
    else:
        st.session_state["history_records"] = records

# 登录与注册功能
def login_page():
    st.markdown('<div class="main-title">🔐 用户登录</div>', unsafe_allow_html=True)
    login_tab, register_tab = st.tabs(["登录", "注册"])
    user = None
    with login_tab:
        username = st.text_input("用户名", key="login_user")
        password = st.text_input("密码", type="password", key="login_pwd")
        if st.button("登录", key="login_btn"):
            user = verify_user(username, password)
            if user:
                st.session_state["user"] = dict(user)
                st.success(f"欢迎，{username}！")
                st.rerun()
            else:
                st.error("用户名或密码错误")
    with register_tab:
        reg_user = st.text_input("新用户名", key="reg_user")
        reg_pwd = st.text_input("新密码", type="password", key="reg_pwd")
        if st.button("注册", key="reg_btn"):
            if reg_user and reg_pwd:
                if add_user(reg_user, reg_pwd):
                    st.success("注册成功，请登录！")
                else:
                    st.error("用户名已存在")
            else:
                st.warning("请填写完整信息")

# 用户管理页面
def page_user_manage():
    st.markdown('<div class="main-title">👤 用户管理</div>', unsafe_allow_html=True)
    users = get_all_users()
    df = pd.DataFrame(users, columns=["id", "用户名", "管理员"])
    st.dataframe(df, hide_index=True)
    del_id = st.number_input("输入要删除的用户ID", min_value=1, step=1, key="del_user_id")
    if st.button("删除用户", key="del_user_btn"):
        delete_user(del_id)
        st.success("删除成功")
        st.rerun()

# 页面定义
def page_history():
    st.markdown('<div class="main-title">📜 历史记录</div>', unsafe_allow_html=True)
    user = st.session_state.get("user")
    if not user:
        st.info("请先登录以查看历史记录")
        return
    records = get_user_records(user["id"])
    if not records:
        st.info("暂无历史记录")
        return
    for i, record in enumerate(records, 1):
        with st.expander(f"记录 #{i} - {record['time']}"):
            if record["type"] == "文本分类":
                st.markdown(f"**类型**: {record['type']}")
                st.markdown(f"**输入文本**: {record['input'][:100]}...")
                color = category_colors.get(record['result'], "#607D8B")
                st.markdown(f"<span style='color:{color}'>**分类结果**: {record['result']}</span>", unsafe_allow_html=True)
                st.markdown(f"**置信度**: {record['confidence']:.2%}" if record['confidence'] else "**置信度**: N/A")
            elif record["type"] == "文件分类":
                st.markdown(f"**类型**: {record['type']}")
                st.markdown(f"**文件名**: {record['filename']}")
                st.markdown(f"**总条数**: {record['total']}")
                st.markdown("**分类结果预览**: ")
                st.dataframe(pd.DataFrame(eval(record['results'])), hide_index=True)
    
    # 历史记录与数据库同步
    if "history_records" not in st.session_state:
        st.session_state["history_records"] = records
    else:
        st.session_state["history_records"] = records

def main():
    # 登录判断
    if "user" not in st.session_state:
        login_page()
        return
    user = st.session_state["user"]
    st.sidebar.image("https://img.icons8.com/color/48/000000/news.png ", width=48)
    st.sidebar.title(f"导航菜单 | 用户：{user['username']}")
    if "sidebar_page" not in st.session_state:
        st.session_state["sidebar_page"] = "首页"
    pages = [
        ("首页", "🏠 首页"),
        ("文本分类", "📝 文本分类"),
        ("文件上传", "📁 文件上传"),
        ("历史记录", "📜 历史记录"),
        ("用户管理", "👤 用户管理") if user.get("is_admin") else None,
        ]
    for p in pages:
        if p:
            key, label = p
            btn = st.sidebar.button(label, key=f"sidebar_btn_{key}")
            if btn:
                st.session_state["sidebar_page"] = key
    page = st.session_state["sidebar_page"]
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
    elif page == "历史记录":
        page_history()
    elif page == "用户管理":
        page_user_manage()

        
    # 登录按钮放
    if st.sidebar.button("退出登录", key="logout_btn"):
        st.session_state.pop("user")
        st.rerun()

if __name__ == '__main__':
    main()