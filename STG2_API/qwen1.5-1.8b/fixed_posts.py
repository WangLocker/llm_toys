import asyncio
import aiohttp
import json

# 替换为你的API密钥和端点
API_KEY = 'your_api_key'
API_URL = 'http://localhost:80/v1/chat/completions'

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# 示例请求体，可以根据实际情况修改
REQUEST_BODY = {
    'model': 'qwen1.5-chat-7b',
    'messages': [{'role': 'user', 'content': '写一段8千字的作文'}],
    'max_tokens': 8192
}

async def fetch(session, payload):
    async with session.post(API_URL, json=payload, headers=HEADERS) as response:
        if response.status == 200:
            result = await response.json()
            return result
        else:
            return {'error': response.status, 'message': await response.text()}

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(20):  # 创建20个并发任务
            tasks.append(fetch(session, REQUEST_BODY))
        responses = await asyncio.gather(*tasks)
        for idx, response in enumerate(responses):
            print(f"Response {idx + 1}: {json.dumps(response, ensure_ascii=False, indent=2)}")

if __name__ == '__main__':
    asyncio.run(main())
