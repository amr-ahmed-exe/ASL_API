"""
Sign Language Recognition API
FastAPI server that receives hand images and returns predicted ASL letters.
"""

import math
import io
import os
import time
import asyncio
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

# Load TFLite Model (2.15 MB) instead of Keras (13 MB) for blazingly fast inference
TFLITE_MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_language_model.tflite")
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Warm-up the interpreter to prevent cold-start latency on the first request
dummy_input = np.zeros((1, 400, 400, 3), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

hd = HandDetector(maxHands=1)
dictionary = enchant.Dict("en_US")

# Cache background image to avoid continuous disk I/O
WHITE_IMG_PATH = os.path.join(SCRIPT_DIR, "white.jpg")
WHITE_IMG = cv2.imread(WHITE_IMG_PATH)

OFFSET = 29


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


# ---------------------------------------------------------------------------
# Core prediction logic (extracted from final_pred.py)
# ---------------------------------------------------------------------------
def predict_letter(cv2image: np.ndarray):
    """
    Given a BGR image containing a hand, detect the hand, draw its skeleton
    on a 400x400 white canvas, run the CNN model, apply post-processing
    rules, and return the predicted letter (or special action).
    Returns (letter: str, status: str)
    """
    hands = hd.findHands(cv2image, draw=False, flipType=True)
    if not hands:
        return None, "no_hand_detected"

    hand = hands[0]
    x, y, w, h = hand["bbox"]
    
    # We already have landmarks from the first detection!
    # No need to run a heavy second ML inference.
    full_pts = hand["lmList"]
    
    # Calculate the bounding box crop origin so we can shift landmarks
    x1 = max(0, x - OFFSET)
    y1 = max(0, y - OFFSET)
    
    # Shift landmarks to be relative to the imaginary cropped image
    pts = []
    for pt in full_pts:
        pts.append([pt[0] - x1, pt[1] - y1])

    white = WHITE_IMG.copy()

    ox = ((400 - w) // 2) - 15
    os1 = ((400 - h) // 2) - 15

    # Draw skeleton lines
    for t in range(0, 4):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + os1),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(5, 8):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + os1),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(9, 12):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + os1),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(13, 16):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + os1),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(17, 20):
        cv2.line(white, (pts[t][0] + ox, pts[t][1] + os1),
                 (pts[t + 1][0] + ox, pts[t + 1][1] + os1), (0, 255, 0), 3)

    cv2.line(white, (pts[5][0] + ox, pts[5][1] + os1),
             (pts[9][0] + ox, pts[9][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[9][0] + ox, pts[9][1] + os1),
             (pts[13][0] + ox, pts[13][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[13][0] + ox, pts[13][1] + os1),
             (pts[17][0] + ox, pts[17][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + ox, pts[0][1] + os1),
             (pts[5][0] + ox, pts[5][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + ox, pts[0][1] + os1),
             (pts[17][0] + ox, pts[17][1] + os1), (0, 255, 0), 3)

    for i in range(21):
        cv2.circle(white, (pts[i][0] + ox, pts[i][1] + os1), 2, (0, 0, 255), 1)

    # -----------------------------------------------------------------------
    # Model inference (TFLite)
    # -----------------------------------------------------------------------
    res = white.reshape(1, 400, 400, 3).astype('float32') # TFLite requires exact dtype
    
    # Ultra-low latency TFLite inference
    interpreter.set_tensor(input_details[0]['index'], res)
    interpreter.invoke()
    prob_tensor = interpreter.get_tensor(output_details[0]['index'])
    
    prob = np.array(prob_tensor[0], dtype="float32")
    
    ch1 = np.argmax(prob, axis=0)
    prob[ch1] = 0
    ch2 = np.argmax(prob, axis=0)
    prob[ch2] = 0
    ch3 = np.argmax(prob, axis=0)

    # -----------------------------------------------------------------------
    # Post-processing rules  (verbatim from final_pred.py)
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
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0]
                and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0]
                and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0]
                and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0]
                and pts[0][0] < pts[20][0]):
            ch1 = 3

    # con for [gh][l]
    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    # con for [l][d]
    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    # con for [l][gh]
    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] < pts[0][0]:
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[1][0] < pts[12][0]:
            ch1 = 4

    # con for [gh][z]
    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1]
                and pts[4][1] + 17 > pts[16][1]
                and pts[4][1] + 17 > pts[20][1]):
            ch1 = 5

    # con for [l][pqz]
    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    # con for [pqz][aemnst]
    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0]
                and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]):
            ch1 = 5

    # con for [pqz][yj]
    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    # con for [l][yj]
    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    # con for [x][yj]
    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    # condition for [x][aemnst]
    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    # condition for [yj][x]
    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    # condition for [c0][x]
    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x]
    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    # con for [x][d]
    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2],
         [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6],
         [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2],
         [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]
                and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [d][pqz]
    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
             and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
                and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]
             and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
                and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    # con for [i][pqz]
    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2],
         [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
                pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 7

    # con for [uvr]
    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
             and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
                and pts[4][1] > pts[14][1]):
            ch1 = 1

    # con for [w]
    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if (not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0]
                 and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0])
                and not (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0]
                         and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0])
                and distance(pts[4], pts[11]) < 50):
            ch1 = 1

    # con for [w]
    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]
                and pts[14][1] > pts[16][1]):
            ch1 = 1

    # -----------------------------------------------------------------------
    # Subgroup classification
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

    if ch1 == 4:
        ch1 = "L"

    if ch1 == 6:
        ch1 = "X"

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
# Suggestions helper
# ---------------------------------------------------------------------------
def get_suggestions(word: str, max_count: int = 4) -> list[str]:
    """
    Return up to max_count word suggestions for the given partial word.

    Strategy:
      1. If the word is a valid dictionary word already → return it alone.
      2. Otherwise get enchant suggestions and keep only those that START WITH
         the typed prefix → "HEL" gives ["HELLO", "HELP", ...] not random words.
      3. If prefix-filtering leaves nothing → fall back to top unfiltered results.
      4. Return [] on any error.
    """
    word = word.strip().upper()
    if not word:
        return []

    try:
        word_lower = word.lower()

        if dictionary.check(word_lower):
            return [word]

        raw = dictionary.suggest(word_lower)

        # Primary: keep suggestions that start with the typed prefix
        filtered = [s.upper() for s in raw if s.lower().startswith(word_lower)]

        # Fallback: unfiltered top results if prefix match returned nothing
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
    return {"status": "running", "message": "Sign Language Recognition API"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_word: str = Form(default=""),
):
    """
    Accepts a hand image + the word built so far, returns the predicted ASL
    letter plus up to 4 word suggestions.

    Multipart form fields:
      - file         : image from camera (JPEG / PNG)
      - current_word : letters typed so far, e.g. "HEL"

    Response:
      {
        "prediction": "L",
        "status": "success",
        "suggestions": ["HELLO", "HELP", "HELD", "HELM"]
      }

    Suggestions always start with current_word (prefix-filtered).
    When current_word is empty, suggestions is [].
    When prediction is " " (space), the mobile app should commit the word.
    """
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        cv2image = np.array(pil_image)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)

        letter, status = predict_letter(cv2image)

        if letter is None:
            return {"prediction": None, "status": status, "suggestions": []}

        # Build suggestions based on the current word + new letter
        suggestions = get_suggestions(current_word)

        return {
            "prediction": letter,
            "status": status,
            "suggestions": suggestions,
        }

    except Exception as e:
        return {
            "prediction": None,
            "status": "error",
            "message": str(e),
            "suggestions": [],
        }


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    Advanced Level 2 WebSocket endpoint: Sign -> Text System.
    Includes Temporal Smoothing, Word Builder, and Spell Correction.
    """
    await websocket.accept()
    print("WebSocket client connected")
    
    loop = asyncio.get_event_loop()
    
    # --- Word Builder State (Level 2 AI) ---
    history = deque(maxlen=7)      # Per-frame smoothing buffer
    current_word = ""              # String builder for the active word
    last_confirmed_letter = None   # Track the last letter added
    consecutive_count = 0          # Stability counter
    REQUIRED_CONSECUTIVE = 5       # Requires 5 identical smoothed frames to confirm letter
    
    last_action_time = time.time()
    WORD_COMMIT_TIMEOUT = 1.5      # Seconds of 'no hand' or 'space' to commit word
    # ---------------------------------------
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Decode image (Fast OpenCV binary decode)
            np_arr = np.frombuffer(data, np.uint8)
            cv2image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv2image is None:
                await websocket.send_json({"prediction": None, "status": "invalid_image"})
                continue
            
            # Predict in a separate thread
            letter, status = await loop.run_in_executor(None, predict_letter, cv2image)
            
            now = time.time()
            word_committed = False
            
            # 1. Handle Timeouts (Auto Commit on inactivity)
            if status == "no_hand_detected":
                if current_word and ((now - last_action_time) > WORD_COMMIT_TIMEOUT):
                    word_committed = True
            
            # 2. Process Detected Letters
            smoothed_letter = None
            confirmed_letter = None
            
            if letter and status == "success":
                history.append(letter)
                # Get the most common letter in the history buffer (Smoothing Layer)
                smoothed_letter = Counter(history).most_common(1)[0][0]
                
                # 3. Stability Checks
                if smoothed_letter == last_confirmed_letter:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                    last_confirmed_letter = smoothed_letter
                
                # 4. Trigger letter confirmation
                if consecutive_count == REQUIRED_CONSECUTIVE:
                    confirmed_letter = smoothed_letter
                    last_action_time = now  # Reset action timer
                    
                    if confirmed_letter == " ":
                        # Explicit Space Gesture confirms word
                        if current_word:
                            word_committed = True
                    else:
                        current_word += confirmed_letter
                        # Clear buffer to require a deliberate pause before typing 
                        # the same letter again (useful for double letters like "HELLO")
                        history.clear()
                        last_confirmed_letter = None
                        consecutive_count = 0
            
            # 5. Spell Check & Commit Word
            final_word = ""
            suggestions = []
            
            if word_committed and current_word:
                # Use enchant for auto-correction context
                suggested_words = get_suggestions(current_word)
                if suggested_words:
                    # Pick the best autocorrect match
                    final_word = suggested_words[0]
                    suggestions = suggested_words
                else:
                    final_word = current_word
                
                # Reset word builder state
                current_word = ""
                history.clear()
                last_confirmed_letter = None
                consecutive_count = 0
                last_action_time = now

            # 6. Stream full language context back to device
            await websocket.send_json({
                "status": status,
                "raw_prediction": letter,               # The raw noisy model output
                "smoothed_letter": smoothed_letter,     # Majority voting output
                "confirmed_letter": confirmed_letter,   # Letter actually added to word
                "current_word": current_word,           # Word currently being built
                "word_committed": word_committed,       # True if a word just finished
                "final_word": final_word,               # The spell-checked finalized word
                "suggestions": suggestions              # Suggestions for UI
            })
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass


