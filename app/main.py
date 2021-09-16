from fastai.imports import *
from fastai.vision import *
from io import BytesIO
import numpy as np

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='./app/static'))

path = Path(__file__).parent

learn = load_learner(path / 'models', 'bb_model.pkl')

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())    

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes)).resize(300)
    pred = learn.predict(img)
    prediction = learn.predict(img)[0]
    confidence = np.array(learn.predict(img)[-1][0]).item()
    return JSONResponse({'result': str(prediction),
                         'confidence': round(confidence * 100, 1),
                        })

if __name__ == '__main__':
    if 'app' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")