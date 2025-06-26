import sqlite3
from contextlib import closing
import hashlib

def get_db_connection():
    conn = sqlite3.connect('news_classifier.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        # 用户表
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )''')
        # 历史记录表
        c.execute('''CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT,
            input TEXT,
            result TEXT,
            confidence REAL,
            filename TEXT,
            total INTEGER,
            results TEXT,
            time TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        # 检查是否存在管理员账号，不存在则创建默认管理员
        c.execute('SELECT * FROM users WHERE is_admin=1')
        admin = c.fetchone()
        # 检查是否存在用户名为admin的账号
        c.execute('SELECT * FROM users WHERE username=?', ('admin',))
        admin_user = c.fetchone()
        if not admin and not admin_user:
            # 默认管理员账号 admin/123456
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      ('admin', hash_password('123456'), 1))
        conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def add_user(username, password, is_admin=0):
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      (username, hash_password(password), is_admin))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def verify_user(username, password):
    hashed = hash_password(password)
    print(f"[DEBUG] 用户名: {username}, 输入密码: {password}, 加密后: {hashed}")
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, is_admin FROM users WHERE username=? AND password=?',
                  (username, hashed))
        user = c.fetchone()
        print(f"[DEBUG] 查询结果: {user}")
        if user:
            return {'id': user[0], 'username': user[1], 'is_admin': user[2]}
        return None

def get_all_users():
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, is_admin FROM users')
        return c.fetchall()

def delete_user(user_id):
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute('DELETE FROM users WHERE id=?', (user_id,))
        conn.commit()

def add_record(user_id, record):
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO records (user_id, type, input, result, confidence, filename, total, results, time)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, record.get('type'), record.get('input'), record.get('result'),
                   record.get('confidence'), record.get('filename'), record.get('total'),
                   str(record.get('results')), record.get('time')))
        conn.commit()

def get_user_records(user_id):
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM records WHERE user_id=? ORDER BY time DESC', (user_id,))
        return c.fetchall()

def get_all_records():
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM records ORDER BY time DESC')
        return c.fetchall()