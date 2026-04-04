"""
Sign Language Recognition API
FastAPI server that receives hand images/skeletons and returns predicted ASL letters.
"""

import math
import io
import os
import time
import asyncio
import json
from collections import Counter, deque
import numpy as np
import cv2
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import enchant

# ---------------------------------------------------------------------------
# Initialize app & load model (runs once at server startup)
# ---------------------------------------------------------------------------
app = FastAPI(title="Sign Language Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load TFLite Model
TFLITE_MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_language_model.tflite")
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Warm-up the interpreter
dummy_input = np.zeros((1, 400, 400, 3), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

# Keep cvzone only for the POST /predict HTTP fallback endpoint
hd = HandDetector(maxHands=1)
dictionary = enchant.Dict("en_US")

WHITE_IMG_PATH = os.path.join(SCRIPT_DIR, "white.jpg")
WHITE_IMG = cv2.imread(WHITE_IMG_PATH)
OFFSET = 29

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

# ---------------------------------------------------------------------------
# Debug counter for skeleton image saving
# ---------------------------------------------------------------------------
_debug_frame_counter = 0

# ---------------------------------------------------------------------------
# Core prediction logic (Skeleton Based)
# ---------------------------------------------------------------------------
def predict_from_skeleton(pts):
    """
    Given 21 points from Flutter, convert to 400x400 canvas,
    run the CNN model, apply post-processing rules, and return predicted letter.
    """
    pts = np.array(pts, dtype=np.float32)

    # 1. تحويل النسب لبيكسلات (نفس القيمة للإتنين عشان منشوهش الإيد!)
    # ملاحظة: الرقم مش مهم لأن Palm-Length Normalization هيظبط الحجم بعدين
    pts *= 640.0

    # 2. المراية متلغية من السيرفر — Flutter بقى بيعكس الإحداثيات صح قبل ما يبعتها

    # 3. التمركز حول نقطة الصفر
    min_x, min_y = pts.min(axis=0)
    pts = pts - [min_x, min_y]

    # -----------------------------------------------------------------------
    # 🌟 Palm-Length Normalization
    # التكبير بناءً على طول كف الإيد (المعصم → قاعدة الأوسط) = ثابت دايماً
    # -----------------------------------------------------------------------
    palm_length = distance(pts[0], pts[9])
    scale = 80.0 / (palm_length + 1e-6)
    pts = pts * scale

    # 4. حساب الأبعاد الجديدة بعد التكبير
    min_x2, min_y2 = pts.min(axis=0)
    max_x2, max_y2 = pts.max(axis=0)
    w_scaled = max_x2 - min_x2
    h_scaled = max_y2 - min_y2

    # 5. التوسيط في الشاشة الـ 400x400
    ox = int((400 - w_scaled) / 2)
    oy = int((400 - h_scaled) / 2)

    pts = pts.astype(int)
    white = WHITE_IMG.copy()

    # Draw skeleton lines
    for t in range(0, 4):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + oy),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + oy), (0, 255, 0), 3)
    for t in range(5, 8):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + oy),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + oy), (0, 255, 0), 3)
    for t in range(9, 12):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + oy),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + oy), (0, 255, 0), 3)
    for t in range(13, 16):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + oy),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + oy), (0, 255, 0), 3)
    for t in range(17, 20):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + oy),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + oy), (0, 255, 0), 3)

    cv2.line(white, (pts[5][0] + ox, pts[5][1] + oy),
             (pts[9][0] + ox, pts[9][1] + oy), (0, 255, 0), 3)
    cv2.line(white, (pts[9][0] + ox, pts[9][1] + oy),
             (pts[13][0] + ox, pts[13][1] + oy), (0, 255, 0), 3)
    cv2.line(white, (pts[13][0] + ox, pts[13][1] + oy),
             (pts[17][0] + ox, pts[17][1] + oy), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + ox, pts[0][1] + oy),
             (pts[5][0] + ox, pts[5][1] + oy), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + ox, pts[0][1] + oy),
             (pts[17][0] + ox, pts[17][1] + oy), (0, 255, 0), 3)

    for i in range(21):
        cv2.circle(white, (pts[i][0] + ox, pts[i][1] + oy), 2, (0, 0, 255), 1)

    # -----------------------------------------------------------------------
    # Inference 
    # -----------------------------------------------------------------------
    res = white.reshape(1, 400, 400, 3).astype('float32') 
    
    # DEBUG: حفظ صورة الـ skeleton كل 30 فريم عشان نشوفها
    global _debug_frame_counter
    _debug_frame_counter += 1
    if _debug_frame_counter % 30 == 1:
        debug_path = os.path.join(SCRIPT_DIR, "debug_skeleton.jpg")
        cv2.imwrite(debug_path, white)
        print(f"[DEBUG] Skeleton image saved to {debug_path}")
    
    interpreter.set_tensor(input_details[0]['index'], res)
    interpreter.invoke()
    prob_tensor = interpreter.get_tensor(output_details[0]['index'])
    prob = np.array(prob_tensor[0], dtype="float32")
    
    # DEBUG: طباعة الاحتمالات عشان نشوف الموديل بيختار إيه
    if _debug_frame_counter % 30 == 1:
        print(f"[DEBUG] Raw probabilities: {prob}")
        print(f"[DEBUG] Top group: {np.argmax(prob)} (confidence: {np.max(prob):.4f})")
    
    ch1 = np.argmax(prob, axis=0)
    prob[ch1] = 0
    ch2 = np.argmax(prob, axis=0)
    prob[ch2] = 0
    # ch3 = np.argmax(prob, axis=0) # Unused in rules
    
    # -----------------------------------------------------------------------
    # Post-processing rules (From Author's final_pred.py)
    # -----------------------------------------------------------------------
    pl = [ch1, ch2]
    # condition for [Aemnst]
    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1],
         [6, 2], [6, 6], [6, 7], [6, 0], [6, 5], [4, 1], [1, 0], [1, 1],
         [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0],
         [2, 6], [4, 6], [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5],
         [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    # condition for [o][s]
    l = [[2, 2], [2, 1]]
    if pl in l:
        if pts[5][0] < pts[4][0]:
            ch1 = 0

    # condition for [c0][aemnst]
    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0]
                and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0]
                and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0]
                and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0]
                and pts[0][0] < pts[20][0]):
            ch1 = 3

    # con for [gh][l]
    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x] - لو الموديل مطلعها X بس هي في الحقيقة L (السبابة مفرودة لفوق)
    l = [[6, 4], [6, 1], [6, 2]]
    if pl in l:
        if pts[8][1] < pts[6][1]:  # السبابة مفرودة = L
            ch1 = 4

    # con for [l][d]
    l = [[1, 4], [1, 6], [1, 1]]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    # con for [l][gh]
    l = [[3, 6], [3, 4]]
    if pl in l:
        if pts[4][0] < pts[0][0]:
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    if pl in l:
        if pts[1][0] < pts[12][0]:
            ch1 = 4

    # con for [gh][z]
    l = [[3, 6], [3, 5], [3, 4]]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3, 2], [3, 1], [3, 6]]
    if pl in l:
        if (pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1]
                and pts[4][1] + 17 > pts[16][1]
                and pts[4][1] + 17 > pts[20][1]):
            ch1 = 5

    # con for [l][pqz]
    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    # con for [pqz][aemnst]
    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    if pl in l:
        if (pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0]
                and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]):
            ch1 = 5

    # con for [pqz][yj]
    l = [[5, 7], [5, 2], [5, 6]]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    # con for [l][yj]
    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    # con for [x][yj]
    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    # condition for [x][aemnst]
    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    # condition for [yj][x]
    l = [[7, 2]]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    # condition for [c0][x]
    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x] - لو الموديل مطلعها L بس هي في الحقيقة X (السبابة متنية زي الخطاف)
    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    if pl in l:
        if pts[8][1] > pts[6][1]:  # السبابة متنية = X
            ch1 = 6

    # con for [x][d]
    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2],
         [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6],
         [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2],
         [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]
                and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [d][pqz]
    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
             and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
                and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
             and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
                and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    # con for [i][pqz]
    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2],
         [7, 5], [7, 1], [7, 6], [7, 7]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
                pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 7

    # con for [uvr]
    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
             and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
                and pts[4][1] > pts[14][1]):
            ch1 = 1

    # con for [w]
    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    if pl in l:
        if (not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0]
                 and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0])
                and not (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0]
                         and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0])
                and distance(pts[4], pts[11]) < 50):
            ch1 = 1

    # con for [w]
    l = [[5, 0], [5, 5], [0, 1]]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1]):
            ch1 = 1

    # -----------------------------------------------------------------------
    # 🛡️ STRICT OVERRIDES FOR MOBILE ANGLES (حماية من الكاميرا الأمامية)
    # -----------------------------------------------------------------------
    # بما إن كاميرا الموبايل بتبص للإيد من تحت، الموديل بيتلخبط.
    # الكود ده بيجبر السيرفر يقرا إشارة L من شكل الصوابع مباشرة:
    
    index_up = pts[8][1] < pts[5][1]       # السبابة مفرودة لفوق
    middle_down = pts[12][1] > pts[9][1]   # الوسطى متنية لتحت
    ring_down = pts[16][1] > pts[13][1]    # البنصر متني لتحت
    pinky_down = pts[20][1] > pts[17][1]   # الخنصر متني لتحت
    thumb_out = abs(pts[4][0] - pts[9][0]) > 30  # الإبهام مفرود لبره

    # لو الإيد واخدة وضع الـ L الحقيقي، افرض النتيجة وامسح أي تخريف فات
    if index_up and middle_down and ring_down and pinky_down and thumb_out:
        ch1 = 4  # المجموعة 4 = L

    # -----------------------------------------------------------------------
    # Subgroup classification (Assigning String Labels)
    # -----------------------------------------------------------------------
    if ch1 == 0:
        ch1 = "S"
        if (pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0]
                and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]):
            ch1 = "A"
        if (pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0]
                and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]
                and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
            ch1 = "T"
        if (pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1]
                and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]):
            ch1 = "E"
        if (pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0]
                and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]):
            ch1 = "M"
        if (pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0]
                and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]):
            ch1 = "N"

    if ch1 == 2:
        if distance(pts[12], pts[4]) > 42:
            ch1 = "C"
        else:
            ch1 = "O"

    if ch1 == 3:
        if distance(pts[8], pts[12]) > 72:
            ch1 = "G"
        else:
            ch1 = "H"

    if ch1 == 7:
        if distance(pts[8], pts[4]) > 42:
            ch1 = "Y"
        else:
            ch1 = "J"

    if ch1 == 4: ch1 = "L"
    if ch1 == 6: ch1 = "X"

    if ch1 == 5:
        if (pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0]
                and pts[4][0] > pts[20][0]):
            if pts[8][1] < pts[5][1]:
                ch1 = "Z"
            else:
                ch1 = "Q"
        else:
            ch1 = "P"

    if ch1 == 1:
        ch1 = "B"  # حرف افتراضي لمنع طباعة رقم 1 لو مفيش شرط اتحقق

        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "B"
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = "D"
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "F"
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "I"
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = "W"
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]
                and pts[4][1] < pts[9][1]):
            ch1 = "K"
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = "U"
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]
                ) and (pts[4][1] > pts[9][1]):
            ch1 = "V"
        if (pts[8][0] > pts[12][0]) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = "R"

    # Space detection
    if ch1 == 1 or ch1 == "E" or ch1 == "S" or ch1 == "X" or ch1 == "Y" or ch1 == "B":
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = " "

    return str(ch1), "success"

# ---------------------------------------------------------------------------
# Fallback logic for POST endpoint (Fallback only)
# ---------------------------------------------------------------------------
def predict_letter(cv2image: np.ndarray):
    hands = hd.findHands(cv2image, draw=False, flipType=True)
    if not hands:
        return None, "no_hand_detected"
    hand = hands[0]
    
    # We pass the points to the new skeleton pipeline
    return predict_from_skeleton(hand["lmList"])

# ---------------------------------------------------------------------------
# Suggestions helper
# ---------------------------------------------------------------------------
def get_suggestions(word: str, max_count: int = 4) -> list[str]:
    word = word.strip().upper()
    if not word: return []
    try:
        word_lower = word.lower()
        if dictionary.check(word_lower):
            return [word]

        raw = dictionary.suggest(word_lower)
        filtered = [s.upper() for s in raw if s.lower().startswith(word_lower)]
        if not filtered:
            filtered = [s.upper() for s in raw]

        return filtered[:max_count]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "running", "message": "Sign Language API"}

@app.get("/debug/skeleton")
def get_debug_skeleton():
    """عشان نشوف صورة الـ skeleton اللي الموديل بيشوفها"""
    from fastapi.responses import FileResponse
    debug_path = os.path.join(SCRIPT_DIR, "debug_skeleton.jpg")
    if os.path.exists(debug_path):
        return FileResponse(debug_path, media_type="image/jpeg")
    return {"error": "No debug image yet. Send a frame first."}

@app.post("/predict")
async def predict(file: UploadFile = File(...), current_word: str = Form(default="")):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        cv2image = np.array(pil_image)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)

        letter, status = predict_letter(cv2image)
        if letter is None:
            return {"prediction": None, "status": status, "suggestions": []}

        suggestions = get_suggestions(current_word)
        return {"prediction": letter, "status": status, "suggestions": suggestions}
    except Exception as e:
        return {"prediction": None, "status": "error", "message": str(e), "suggestions": []}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected (Skeleton Text Mode)")
    
    loop = asyncio.get_event_loop()
    
    history = deque(maxlen=7)      
    current_word = ""              
    last_confirmed_letter = None   
    last_appended_letter = None    
    consecutive_count = 0          
    REQUIRED_CONSECUTIVE = 5       
    missing_frames = 0             # 👈 العداد الجديد لمنع الفليكر
    
    last_action_time = time.time()
    WORD_COMMIT_TIMEOUT = 1.5      
    
    try:
        while True:
            data_text = await websocket.receive_text()
            pts = json.loads(data_text)
            
            if not pts or len(pts) != 21:
                status = "no_hand_detected"
                letter = None
            else:
                letter, status = await loop.run_in_executor(None, predict_from_skeleton, pts)
            
            now = time.time()
            word_committed = False
            
            # --- المعالجة الجديدة لاختفاء الإيد ---
            if status == "no_hand_detected":
                missing_frames += 1
                # لو الإيد اختفت لأكتر من 5 فريمات (حوالي نص ثانية)، هنصفر الحرف عشان لو عاوز تكتب LL
                if missing_frames > 5:
                    last_appended_letter = None 
                    
                # لو عدى ثانية ونص، بنقفل الكلمة ونجيب الاقتراحات
                if current_word and ((now - last_action_time) > WORD_COMMIT_TIMEOUT):
                    word_committed = True
            else:
                missing_frames = 0  # تصفير العداد أول ما الإيد تظهر
                
            smoothed_letter = None
            confirmed_letter = None
            
            # لو لقطنا حرف
            if letter and status == "success":
                history.append(letter)
                smoothed_letter = Counter(history).most_common(1)[0][0]
                
                if smoothed_letter == last_confirmed_letter:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                    last_confirmed_letter = smoothed_letter
                
                # لما نتأكد من الحرف بعد 5 فريمات ثبات
                if consecutive_count == REQUIRED_CONSECUTIVE:
                    confirmed_letter = smoothed_letter
                    last_action_time = now  
                    
                    if confirmed_letter == " ":
                        if current_word:
                            word_committed = True
                        last_appended_letter = " "
                    else:
                        # هنضيف الحرف للكلمة بس لو كان مختلف عن آخر حرف اتضاف
                        if confirmed_letter != last_appended_letter:
                            current_word += confirmed_letter
                            last_appended_letter = confirmed_letter
                            
                    history.clear()
                    last_confirmed_letter = None
                    consecutive_count = 0
            
            final_word = ""
            suggestions = []

            # اقتراحات حية وهي الكلمة بتتكتب
            if current_word and len(current_word) >= 2:
                suggestions = get_suggestions(current_word)
            
            if word_committed and current_word:
                suggested_words = get_suggestions(current_word)
                if suggested_words:
                    final_word = suggested_words[0]
                    suggestions = suggested_words
                else:
                    final_word = current_word
                
                current_word = ""
                history.clear()
                last_confirmed_letter = None
                last_appended_letter = None
                consecutive_count = 0
                last_action_time = now

            await websocket.send_json({
                "status": status,
                "raw_prediction": letter,               
                "smoothed_letter": smoothed_letter,     
                "confirmed_letter": confirmed_letter,   
                "current_word": current_word,           
                "word_committed": word_committed,       
                "final_word": final_word,               
                "suggestions": suggestions              
            })
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

