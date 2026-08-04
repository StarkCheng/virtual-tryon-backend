from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import base64
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
IMAGE_MODEL = os.getenv('IMAGE_MODEL', 'gpt-image-2')
RESTORE_FACE = os.getenv('RESTORE_FACE', 'false').lower() == 'true'

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


def bytes_to_cv2(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def restore_original_face(original_bytes, generated_bytes):
    """把原圖人臉羽化貼回生成圖(可選的保險機制)"""
    try:
        orig = bytes_to_cv2(original_bytes)
        gen = bytes_to_cv2(generated_bytes)
        if orig is None or gen is None:
            return generated_bytes

        if orig.shape[:2] != gen.shape[:2]:
            orig = cv2.resize(orig, (gen.shape[1], gen.shape[0]))

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        if len(faces) == 0:
            print('No face detected, skipping face restoration')
            return generated_bytes

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x, pad_y = int(w * 0.25), int(h * 0.35)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(orig.shape[1], x + w + pad_x), min(orig.shape[0], y + h + pad_y)

        mask = np.zeros(orig.shape[:2], dtype=np.float32)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        blur_size = max(21, (min(axes) // 4) * 2 + 1)
        mask = np.clip(cv2.GaussianBlur(mask, (blur_size, blur_size), 0), 0, 1)
        mask3 = cv2.merge([mask, mask, mask])

        blended = (orig.astype(np.float32) * mask3 +
                   gen.astype(np.float32) * (1 - mask3)).astype(np.uint8)

        success, encoded = cv2.imencode('.png', blended)
        if success:
            print(f'Face restored at ({x1},{y1})-({x2},{y2})')
            return encoded.tobytes()
        return generated_bytes

    except Exception as e:
        print(f'Face restoration skipped: {type(e).__name__}: {e}')
        return generated_bytes


def openai_images_edit(files, data):
    """呼叫 OpenAI images/edits,回傳 base64 圖片字串"""
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    response = requests.post(
        'https://api.openai.com/v1/images/edits',
        headers=headers,
        files=files,
        data=data,
        timeout=600,
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
    print(f'OpenAI API error: {response.status_code} {response.text[:800]}')
    raise HTTPException(status_code=500, detail=f'OpenAI error {response.status_code}: {response.text[:200]}')


@app.get('/health')
async def health():
    return {'status': 'ok', 'model': IMAGE_MODEL, 'restore_face': RESTORE_FACE}


@app.post('/api/tryon')
async def generate_tryon(request: TryOnRequest):
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail='OpenAI API key not configured')

        body_bytes = data_url_to_bytes(request.body_image)
        clothing_bytes = data_url_to_bytes(request.clothing_image)

        files = [
            ('image[]', ('body.png', body_bytes, 'image/png')),
            ('image[]', ('clothing.png', clothing_bytes, 'image/png')),
        ]
        data = {
            'model': IMAGE_MODEL,
            'prompt': (
                'The first image is a photo of a person. The second image is a clothing item. '
                'Generate a photorealistic image of the SAME person from the first image wearing '
                'the clothing item from the second image.\n\n'
                'ABSOLUTE REQUIREMENTS - the following must be preserved pixel-perfectly from the first image:\n'
                '1. FACE: identical facial features, identical facial structure, identical expression, '
                'identical eyes, nose, mouth, eyebrows, identical skin tone. Zero distortion. '
                'The face must look like the exact same individual - not a similar-looking person.\n'
                '2. HAIR: identical hairstyle, length, colour and how it falls.\n'
                '3. BODY: identical body proportions, height and build.\n'
                '4. POSE: identical body position, limb placement and camera angle.\n'
                '5. BACKGROUND: identical background, lighting direction and colour temperature.\n\n'
                'ONLY the clothing changes. Fit the garment naturally to the body with realistic '
                'drape, folds and shadows consistent with the original lighting. '
                'Preserve the garment\'s exact colour, pattern, texture and design details '
                'from the second image.'
            ),
            'input_fidelity': 'high',
            'quality': 'high',
        }

        img_b64 = openai_images_edit(files, data)

        if RESTORE_FACE:
            generated_bytes = base64.b64decode(img_b64)
            restored_bytes = restore_original_face(body_bytes, generated_bytes)
            img_b64 = base64.b64encode(restored_bytes).decode('utf-8')

        return {'success': True, 'image': f'data:image/png;base64,{img_b64}', 'model': IMAGE_MODEL}

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

        image_bytes = data_url_to_bytes(request.image)

        files = {'image': ('image.png', image_bytes, 'image/png')}
        data = {
            'model': IMAGE_MODEL,
            'prompt': (
                f'{request.prompt}\n\n'
                'ABSOLUTE REQUIREMENT: The person\'s face must remain pixel-perfectly identical '
                'to the input image - same facial features, same structure, same expression, '
                'same skin tone. Zero facial distortion or alteration. '
                'It must remain recognisably the exact same individual. '
                'Also preserve their hairstyle, body proportions and pose unless the instruction '
                'explicitly asks otherwise.'
            ),
            'input_fidelity': 'high',
            'quality': 'high',
        }

        img_b64 = openai_images_edit(files, data)

        if RESTORE_FACE:
            generated_bytes = base64.b64decode(img_b64)
            restored_bytes = restore_original_face(image_bytes, generated_bytes)
            img_b64 = base64.b64encode(restored_bytes).decode('utf-8')

        return {'success': True, 'image': f'data:image/png;base64,{img_b64}'}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Refinement failed: {type(e).__name__}: {str(e)}')


@app.get('/')
async def root():
    return {'message': 'Virtual Try-On Backend API', 'status': 'running', 'model': IMAGE_MODEL}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)