import json

from flask import Flask, request, make_response
from flask_cors import CORS
import os
from datetime import datetime
from mmocr.utils.ocr import MMOCR

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


@app.route("/recognition", methods=["GET"])
def recognition():
    result = make_response()
    result.headers['Content-Type'] = "application/json"
    if not request.args.get('src'):
        # init
        result.response = json.dumps({'code': 0, 'count': 0, 'data': []})
    else:
        # recogntion
        src = request.args.get('src')
        lan = request.args.get('lan')
        t = str(datetime.timestamp(datetime.now()))
        out_file = IMAGES_PATH + t + '.jpg'
        if lan == 'eng':
            # english
            config = 'configs/textrecog/svtr/svtr_academic_dataset.py'
            checkpoint = 'E:/Recongntion/mmocr-main/work_dirs/svtr_eng/svtrv1/epoch_18.pth'
        else:
            # chinese
            config = 'configs/textrecog/svtr/svtr_academic_dataset_chinese.py'
            checkpoint = 'E:/Recongntion/mmocr-main/work_dirs/svtr_chinese/svtr-new/epoch_98.pth'
        mmocr = MMOCR(det=None, recog='SVTR', recog_config=config, recog_ckpt=checkpoint)
        pp_result = mmocr.readtext(src, print_result=False, output=out_file)

        # score cal
        if lan == 'eng':
            score = pp_result[0]['score']
        else:
            score = sum(pp_result[0]['score']) / len(pp_result[0]['score'])

        # result construction
        result.response = json.dumps(
            {'code': 0, 'count': 1, 'data': [
                {'score': score, 'recresult': pp_result[0]['text'], 'file': out_file}]})
    return result


if __name__ == "__main__":
    app.run(port=5002, host="127.0.0.1", debug=False)
