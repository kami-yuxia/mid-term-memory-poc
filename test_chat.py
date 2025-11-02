import requests
import json

# API 端点
url = "http://127.0.0.1:8000/chat"

# 请求数据
data = {
    "message": "我刚刚问了你什么问题？你复述一下就好了.",
    "session_id": "conversation_002"
}

# 发送请求
try:
    response = requests.post(
        url, 
        headers={"Content-Type": "application/json"}, 
        data=json.dumps(data),
        timeout=30
    )
    
    # 打印响应
    print(f"状态码: {response.status_code}")
    print(f"响应头: {response.headers}")
    print(f"响应内容: {response.text}")
    
    # 如果响应是 JSON 格式，解析并打印
    if response.headers.get('content-type', '').startswith('application/json'):
        result = response.json()
        print(f"解析后的 JSON: {result}")
    else:
        print("响应不是 JSON 格式")
        
except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")
except json.JSONDecodeError as e:
    print(f"JSON 解析错误: {e}")
