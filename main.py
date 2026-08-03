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
TRYON_ENGINE = os.getenv('TRYON_ENGINE', 'openai')  # 'replicate' 或 'openai'

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


def data_url_to_bytes(data_url: str) -> bytes:
    """把前端傳來的 base64 data URL 轉成圖片位元組"""
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    return base64.b64decode(data_url)


def openai_images_edit(files, data):
    """呼叫 OpenAI images/edits,回傳 base64 圖片字串"""
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    response = requests.post(
        'https://api.openai.com/v1/images/edits',
        headers=headers,
        files=files,
        data=data,
        timeout=300,
    )
    if response.status_code == 200:
        result = response.json()
        if 'data' in result and len(result['data']) > 0:
            item = result['data'][0]
            if 'b64_json' in item:
                return item['b64_json']
            if 'url' in item:
                img_response = requests.get(item['url'])
                if img_response.status_code == 200:
                    return base64.b64encode(img_response.content).decode('utf-8')
    print(f'OpenAI API error: {response.status_code} {response.text[:500]}')
    raise HTTPException(status_code=500, detail=f'OpenAI error {response.status_code}: {response.text[:200]}')


@app.get('/health')
async def health():
    return {'status': 'ok', 'tryon_engine': TRYON_ENGINE}


@app.post('/api/tryon')
async def generate_tryon(request: TryOnRequest):
    try:
        if TRYON_ENGINE == 'openai':
            return tryon_with_openai(request)
        return tryon_with_replicate(request)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Try-on generation failed: {type(e).__name__}: {str(e)}')


def tryon_with_replicate(request: TryOnRequest):
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

    if isinstance(output, list):
        output = output[0] if output else None

    if output is None:
        raise HTTPException(status_code=500, detail='Model returned no output')

    result_url = str(output)
    response = requests.get(result_url)
    if response.status_code == 200:
        img_base64 = base64.b64encode(response.content).decode('utf-8')
        return {'success': True, 'image': f'data:image/png;base64,{img_base64}', 'engine': 'replicate'}

    raise HTTPException(status_code=500, detail=f'Failed to download result: {response.status_code}')


def tryon_with_openai(request: TryOnRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail='OpenAI API key not configured')

    body_bytes = data_url_to_bytes(request.body_image)
    clothing_bytes = data_url_to_bytes(request.clothing_image)

    files = [
        ('image[]', ('body.png', body_bytes, 'image/png')),
        ('image[]', ('clothing.png', clothing_bytes, 'image/png')),
    ]
    data = {
        'model': 'gpt-image-1',
        'prompt': (
            'Make the person in the first image wear the clothing item shown in the second image. '
            'Keep the person\'s face, hair, body shape, pose and the background exactly the same. '
            'Only change their outfit to the new clothing. Make it look natural and realistic.'
        ),
        'input_fidelity': 'high',
    }

    img_b64 = openai_images_edit(files, data)
    return {'success': True, 'image': f'data:image/png;base64,{img_b64}', 'engine': 'openai'}


@app.post('/api/refine')
async def refine_image(request: RefineRequest):
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail='OpenAI API key not configured')

        image_bytes = data_url_to_bytes(request.image)

        files = {
            'image': ('image.png', image_bytes, 'image/png'),
        }
        data = {
            'model': 'gpt-image-1',
            'prompt': request.prompt,
        }

        img_b64 = openai_images_edit(files, data)
        return {'success': True, 'image': f'data:image/png;base64,{img_b64}'}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Refinement failed: {type(e).__name__}: {str(e)}')


@app.get('/')
async def root():
    return {'message': 'Virtual Try-On Backend API', 'status': 'running', 'tryon_engine': TRYON_ENGINE}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)