from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from io import BytesIO
from pathlib import Path
from fastai.vision import (
    load_learner,
    open_image,
)
import sys
import uvicorn
import aiohttp
import asyncio


model_file_name = 'models'
classes = ['orca', 'dolphin', 'octopus']
path = Path(__file__).parent

app = Starlette(debug=True)
app.mount('/static', StaticFiles(directory='static'))


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    # load the export.pkl from models directory
    return load_learner(model_file_name)


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    res = str(learn.predict(img)[0])
    return JSONResponse({'result': res})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=9000)
