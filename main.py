from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import replicate
import requests
import os
import base64
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
FRONTEND_URL = os.getenv('FRONTEND_URL', '*')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, 'http://localhost:3000', 'https://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/api/tryon')
async def generate_tryon(body_image: str, clothing_image: str):
    """
    生成虛擬試衣效果
    輸入：base64 格式的身體圖片和衣服圖片
    輸出：base64 格式的試穿結果
    """
    try:
        if not REPLICATE_API_TOKEN:
            raise HTTPException(status_code=500, detail='Replicate API token not configured')
        
        client = replicate.Replicate(api_token=REPLICATE_API_TOKEN)
        
        # 使用 IDM-VTON 模型進行虛擬試衣
        output = client.run(
            'cuuupid/idm-vton',
            input={
                'cloth_image': clothing_image,
                'human_image': body_image,
            }
        )
        
        # 返回生成結果的 URL
        if output and isinstance(output, list) and len(output) > 0:
            result_url = output[0]
            
            # 下載圖片轉換為 base64
            response = requests.get(result_url)
            if response.status_code == 200:
                img_base64 = base64.b64encode(response.content).decode('utf-8')
                return {'success': True, 'image': f'data:image/png;base64,{img_base64}'}
        
        raise HTTPException(status_code=500, detail='Failed to generate try-on image')
    
    except Exception as e:
        print(f'Error in generate_tryon: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Try-on generation failed: {str(e)}')

@app.post('/api/refine')
async def refine_image(image: str, prompt: str):
    """
    使用 GPT Image 1.5 編輯圖片
    """
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail='OpenAI API key not configured')
        
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # 調用 OpenAI 圖像編輯 API (gpt-image-1)
        payload = {
            'model': 'gpt-image-1',
            'image': image,
            'prompt': prompt,
        }
        
        response = requests.post(
            'https://api.openai.com/v1/images/edit',
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                img_url = result['data'][0]['url']
                
                # 下載圖片轉換為 base64
                img_response = requests.get(img_url)
                if img_response.status_code == 200:
                    img_base64 = base64.b64encode(img_response.content).decode('utf-8')
                    return {'success': True, 'image': f'data:image/png;base64,{img_base64}'}
        
        raise HTTPException(status_code=500, detail='Image refinement failed')
    
    except Exception as e:
        print(f'Error in refine_image: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Refinement failed: {str(e)}')

@app.get('/')
async def root():
    return {'message': 'Virtual Try-On Backend API', 'status': 'running'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)