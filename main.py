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
            'cuuupid/idm-vton:139cb1163486954531b765d4ac3bb6d3e02fe121151665adfc3b47e9ba3ebf67',
            input={
                'garm_img': request.clothing_image,
                'human_img': request.body_image,
                'garment_des': 'clothing',
            }
        )

        # 模型可能回傳列表或單一檔案物件，兩種都處理
        if isinstance(output, list):
            output = output[0] if output else None

        if output is None:
            raise HTTPException(status_code=500, detail='Model returned no output')

        result_url = str(output)
        response = requests.get(result_url)
        if response.status_code == 200:
            img_base64 = base64.b64encode(response.content).decode('utf-8')
            return {'success': True, 'image': f'data:image/png;base64,{img_base64}'}

        raise HTTPException(status_code=500, detail=f'Failed to download result: {response.status_code}')

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Try-on generation failed: {type(e).__name__}: {str(e)}')


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
                data_item = result['data'][0]
                if 'b64_json' in data_item:
                    return {'success': True, 'image': f"data:image/png;base64,{data_item['b64_json']}"}
                if 'url' in data_item:
                    img_response = requests.get(data_item['url'])
                    if img_response.status_code == 200:
                        img_base64 = base64.b64encode(img_response.content).decode('utf-8')
                        return {'success': True, 'image': f'data:image/png;base64,{img_base64}'}

        print(f'OpenAI API error: {response.status_code} {response.text[:500]}')
        raise HTTPException(status_code=500, detail=f'Image refinement failed: {response.status_code}')

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Refinement failed: {type(e).__name__}: {str(e)}')


@app.get('/')
async def root():
    return {'message': 'Virtual Try-On Backend API', 'status': 'running'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)