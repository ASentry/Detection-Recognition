import json

from flask import Flask, request, make_response
from flask_cors import CORS
import os
from datetime import datetime

import mmcv
from mmdet.apis import init_detector

from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401

app = Flask(__name__)
CORS(app, supports_credentials=True)
IMAGES_PATH = "G:/DetRecSys/images/"


@app.route("/upload", methods=["POST"])
def save_img():
    img = request.files.get('file')
    result = make_response()
    if not os.path.exists(IMAGES_PATH):
        os.mkdir(IMAGES_PATH)
    t = str(datetime.timestamp(datetime.now()))
    savepath = IMAGES_PATH + t + '.jpg'
    img.save(savepath)
    result.headers['Content-Type'] = "application/json"
    result.response = json.dumps({'file': savepath})
    return result


@app.route("/detection", methods=["POST"])
def detection():
    src = request.get_json(force=True)
    img = src['src']
    config = 'configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py'
    checkpoint = 'F:/fcenet-new-model/ctw1500-transv3/epoch_1117org.pth'
    device = 'cuda:0'
    t = str(datetime.timestamp(datetime.now()))
    out_file = IMAGES_PATH + t + '.jpg'
    model = init_detector(config, checkpoint, device=device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    # test a single image
    result = model_inference(model, img)
    # print(f'result: {result}')

    # show the results
    img = model.show_result(
        img, result, thickness=6, out_file=out_file, show=False)

    if img is None:
        img = mmcv.imread(img)

    mmcv.imwrite(img, out_file)
    result = make_response()
    result.headers['Content-Type'] = "application/json"
    result.response = json.dumps({'file': out_file})
    return result


if __name__ == "__main__":
    app.run(port=5001, host="127.0.0.1", debug=False)
