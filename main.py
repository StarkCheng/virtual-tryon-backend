from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import replicate
import requests
import os
import base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
FRONTEND_URL = os.getenv('FRONTEND_URL', '*')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class TryOnRequest(BaseModel):
    body_image: str
    clothing_image: str

class RefineRequest(BaseModel):
    image: str
    prompt: str

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/api/tryon')
async def generate_tryon(request: TryOnRequest):
    try:
        if not REPLICATE_API_TOKEN:
            raise HTTPException(status_code=500, detail='Replicate API token not configured')
        
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        
        output = client.run(
            'cuuupid/idm-vton',
            input={
                'cloth_image': request.clothing_image,
                'human_image': request.body_image,
            }
        )
        
        if output and isinstance(output, list) and len(output) > 0:
            result_url = output[0]
            response = requests.get(result_url)
            if response.status_code == 200:
                img_base64 = base64.b64encode(response.content).decode('utf-8')
                return {'success': True, 'image': f'data:image/png;base64,{img_base64}'}
        
        raise HTTPException(status_code=500, detail='Failed to generate try-on image')
    
    except Exception as e:
        print(f'Error in generate_tryon: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Try-on generation failed: {str(e)}')

@app.post('/api/refine')
async def refine_image(request: RefineRequest):
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail='OpenAI API key not configured')
        
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gpt-image-1',
            'image': request.image,
            'prompt': request.prompt,
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