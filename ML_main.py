#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Intel社のサンプルを元にy.fukuharaが簡略化と日本語コメントの追記（2022/07）
from time import sleep

import logging as log
import sys

from fastapi import FastAPI
import time
from pydantic import BaseModel

import cv2
import io

from PIL import Image
from io import BytesIO


import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type
import base64
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


def ML_main(img):
    #log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    model_path_age = "./intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
    model_path_emotion = ".\intel\emotions-recognition-retail-0003\FP32\emotions-recognition-retail-0003.xml"
    image_path = img

    #初期化
    #log.info('Creating OpenVINO Runtime Core')
    core = Core()


    ## ageモデルの読み込み
    #log.info(f'ageモデルの読み込み: {model_path_age}')
    # .xmlを読み込む
    model_age = core.read_model(model_path_age)

    ## emotionモデルの読み込み
    #log.info(f'emotionモデルの読み込み: {model_path_emotion}')
    # .xmlを読み込む
    model_emotion = core.read_model(model_path_emotion)

    # 画像の読み込み
    image = cv2.imread(image_path)
    # 「N」の次元を追加 
    input_tensor = np.expand_dims(image, 0)



    #----age処理-------------------------------------------------------------------

    ##前処理
    pppa = PrePostProcessor(model_age)

    # 入力データの情報（高さ、幅）
    _, h, w, _ = input_tensor.shape
    #log.info("画像の高さ: %d px, 幅: %d px", h, w)
    
    # 1) 入力データ情報をセットする
    # 先ほどのデータ形状や、データ形式、データの並びなど
    # 画像の場合は以下の形式でほとんどの場合OK        
    pppa.input().tensor() \
        .set_shape(input_tensor.shape) \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))

    # 2) 前処理の追加    
    # - ここでは入力画像をモデルへの入力サイズに合った大きさに変換している。
    pppa.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    # 3) データの並びをモデルに合った形に並べ替える
    pppa.input().model().set_layout(Layout('NCHW'))

    # 4) モデルに前処理プロセスを適用する
    model_age = pppa.build()

    #log.info('モデルをCPU上で計算できるように準備')
    compiled_model = core.compile_model(model_age, "CPU")

    
    #log.info('推論の実行')
    results = compiled_model.infer_new_request({0: input_tensor})

    predictions = list(results.values())
    age =  predictions[1].reshape(-1)

    #print( "--------age ==", age[0]*100)

    #--------emotionモデル処理--------------------------------------------------
    pppe = PrePostProcessor(model_emotion)

    # 入力データの情報（高さ、幅）
    _, h, w, _ = input_tensor.shape
    #log.info("画像の高さ: %d px, 幅: %d px", h, w)    

    # 1) 入力データ情報をセットする
    # 先ほどのデータ形状や、データ形式、データの並びなど
    # 画像の場合は以下の形式でほとんどの場合OK    
    pppe.input().tensor() \
        .set_shape(input_tensor.shape) \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))

    # 2) 前処理の追加
    # - ここでは入力画像をモデルへの入力サイズに合った大きさに変換している。
    pppe.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    # 3) データの並びをモデルに合った形に並べ替える
    pppe.input().model().set_layout(Layout('NCHW'))

    # 4) モデルに前処理プロセスを適用する
    model_emotion = pppe.build()

    # --------------------------- Step 5. モデルをCPUにロード ------------------------
    #log.info('モデルをCPU上で計算できるように準備')
    compiled_model = core.compile_model(model_emotion, "CPU")

    # --------------------------- Step 6. 推論を実行する -----------------------------
    #log.info('推論の実行')
    results = compiled_model.infer_new_request({0: input_tensor})

    # --------------------------- Step 7. 結果の出力 ---------------------------------
    predictions = list(results.values())
    # 最初の出力データを行ベクトルにする    
    emotion = predictions[0].reshape(-1)
    # 結果の表示
    #print(" neutral    happy      sad        surprise   anger")
    #print(emotion)
    emotion_t = (emotion > 0.4)

    age_alert = False
    emotion_alert = False


    
    #results_list = [age[0]*100 , emotion_t[2],emotion_t[3],emotion_t[4]]

    if age[0]*100 >= 75:
        age_alert = True

    if emotion_t[2] == True or emotion_t[3] == True or emotion_t[4] == True:
        emotion_alert = True

    results_list = [age_alert ,emotion_alert,age[0]*100,emotion[2],emotion[3],emotion[4]]

    return results_list

# if __name__ == '__main__':
#     data = 'data'

#     img = base64.b64decode(data.encode())
    
#     result_a = ML_main(img)
#     #print(result_a)
#     sys.exit()

class image_(BaseModel):
    value: str

app = FastAPI()
#origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/apiv2/{hogeh}")
async def read_root(hogeh):
    print("------activate-----------------------------")
    
    #print("bases ---------------",hogeh)
    data1 = hogeh.replace("_s-","/")
    data1 = data1.replace("_c-",":")
    data1 = data1.replace("_sc-",";")
    data1 = data1.replace("_cn-",",")
    data1 = data1.replace("_p-","+")

    print(data1)
    
    data1 += "=" * ((4 - len(data1) % 4) % 4)

    #data2 = base64.b64encode(bytes(data1,'utf-8'))
    #print(type(data))
    
    
    

    # fh = open("Input.png","wb")
    # fh.write(data1.decode('png'))
    # fh.close()
    

    #print("data ---------------",data2)

    #------------png--------------------------------------------------------#
    #img_by = base64.b64decode(data1.encode())

    #img_by = bytes( "b'"+ data1 +"'" )
    img_by = data1
    print(img_by)

    #img_by = base64.b64encode(bytes(data1,'utf-8'))


    #print(data1,"--------------------------------------------")
    #img_by = base64.b64encode(bytes(data1))

  
    with open("input.png",mode="wb") as f4:
        img_de = base64.b64decode(img_by)
        print("hogehoge")

        
        
        f4.write(img_de)

    #plan2
    # im = Image.open(BytesIO(base64.b64decode(data1)))
    # im.save("input.png",'PNG')

    #plan3





    #-------------jpg---------------------------------------#
    #jpg = np.frombuffer(img_by,dtype=np.uint8)
    #img_path = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    #input_path _ r"test.jpg""
    #cv2.imwrite(input_path,img_path)

    #------------------------------------------------------#

    
    #img_path = base64.b64decode(data.encode())
    #print("img_path========================",img_path)
    result = ML_main('input.png')
    time.sleep(1)
    print("----------complete-----------")
    return {"index1":result[0],"index2":result[1],"age":result[2],"sad":result[3],"surprise":result[4],"anger":result[5]}

@app.get("/hello")
async def helloo():
    return("hello")