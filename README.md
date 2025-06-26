# 新闻分类项目

## 项目描述
这是一个基于机器学习的新闻分类系统，使用Python实现。

## 文件结构
- `data.py`: 数据处理脚本
- `database.py`: 数据库操作模块
- `evaluate.py`: 模型评估脚本
- `model.py`: 机器学习模型实现
- `news_classifier.db`: SQLite数据库文件
- `stacking_news_model.pkl`: 训练好的模型文件
- `streamlit_app.py`: 基于Streamlit的Web应用

## 使用说明
1. 安装依赖: `pip install -r requirements.txt`
2. 运行Web应用: `streamlit run streamlit_app.py`

## 功能
- 新闻文本分类
- 模型训练与评估
- 基于Web的交互界面