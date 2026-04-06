import os
import json
import asyncio
from collections import Counter, deque
import numpy as np
import torch
import enchant
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from train_model import SignLanguageAttentionModel

# ---------------------------------------------------------------------------
# 1. إعداد السيرفر والمكتبات
# ---------------------------------------------------------------------------
app = FastAPI(title="ASL Recognition API (PyTorch Transformer)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dictionary = enchant.Dict("en_US")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 2. تحميل الموديل (PyTorch Transformer)
# ---------------------------------------------------------------------------
print("Loading PyTorch Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageAttentionModel(num_classes=24).to(device)

MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_language_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# الحروف اللي الموديل متدرب عليها (مفيش J ومفيش Z)
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
          'T', 'U', 'V', 'W', 'X', 'Y']

# ---------------------------------------------------------------------------
# 3. دالة التوقع الأساسية
# ---------------------------------------------------------------------------
def predict_from_pytorch(pts, j_z_history=None):
    flat_landmarks = []
    
    # تحويل النقط لـ 63 رقم خام (X, Y, Z)
    for p in pts:
        x = float(p[0])
        y = float(p[1])
        z = float(p[2]) if len(p) > 2 else 0.0 
        flat_landmarks.extend([x, y, z])
        
    if len(flat_landmarks) != 63:
        return None, "invalid_landmarks_shape"

    # التوقع باستخدام الـ Transformer
    landmarks_tensor = torch.tensor(flat_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
    letter = LABELS[predicted_idx.item()]
    
    # ── J/Z Motion Detection ──────────────────────────────────────────
    # الموديل مش متدرب على J و Z (حروف حركية) — بنكتشفهم من الـ motion path
    if j_z_history and len(j_z_history) >= 8:

        # ▶ J — حركة الخنصر: ينزل لأسفل ويعمل خطاف (ي)
        # pts[20] = pinky tip
        if letter == "I":
            pinky = [h[1] for h in j_z_history]
            y_displacement = pinky[-1][1] - pinky[0][1]   # موجب = نزول
            x_displacement = abs(pinky[-1][0] - pinky[0][0])

            # نتحقق إن الخنصر اتحرك للأسفل بشكل واضح مع انحناء لليمين/يسار
            if y_displacement > 0.08 and x_displacement > 0.025:
                letter = "J"

        # ▶ Z — حركة السبابة: يمين → أسفل-يسار → يمين (تغيير اتجاه في X مرتين)
        # pts[8] = index finger tip
        elif letter in ("D", "U", "V"):
            index = [h[0] for h in j_z_history]

            # نحسب velocity في X لكل إطار متتالي
            x_vels = [index[i][0] - index[i-1][0] for i in range(1, len(index))]

            # نعد كام مرة الاتجاه اتغير (right→left أو left→right)
            direction_changes = 0
            cur_dir = None
            for v in x_vels:
                if abs(v) > 0.005:           # تجاهل الحركات الصغيرة جداً
                    new_dir = 1 if v > 0 else -1
                    if cur_dir is not None and new_dir != cur_dir:
                        direction_changes += 1
                    cur_dir = new_dir

            # Z يحتاج ≥ 2 تغيير اتجاه + حركة Y (الخط المائل) + حركة X كلية
            y_total = abs(index[-1][1] - index[0][1])
            x_total = abs(index[-1][0] - index[0][0])
            if direction_changes >= 2 and y_total > 0.04 and x_total > 0.03:
                letter = "Z"
    
    return letter, "success"

# ---------------------------------------------------------------------------
# 4. دالة الاقتراحات (Spell Check)
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
# 5. الـ WebSocket (للاتصال المباشر مع الموبايل)
# ---------------------------------------------------------------------------
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("Flutter Client Connected!")

    loop = asyncio.get_running_loop()
    
    # متغيرات الاستقرار وتكوين الجمل
    history = deque(maxlen=7)
    j_z_history = deque(maxlen=15)

    current_word = ""
    current_sentence = ""
    last_confirmed_letter = None
    consecutive_count = 0
    REQUIRED_CONSECUTIVE = 5

    # cache للاقتراحات — بيتحدث بس لما current_word يتغير
    last_word_for_suggestions = ""
    cached_suggestions: list = []

    try:
        while True:
            data_text = await websocket.receive_text()

            # أوامر التحكم من الموبايل
            if data_text == "CLEAR":
                current_word = ""
                current_sentence = ""
                history.clear()
                j_z_history.clear()
                last_confirmed_letter = None
                consecutive_count = 0
                continue
            
            if data_text.startswith("COMMIT:"):
                selected_word = data_text.split(":", 1)[1]
                current_sentence += (" " + selected_word) if current_sentence else selected_word
                current_word = ""
                history.clear()
                continue

            # استلام النقط
            try:
                pts = json.loads(data_text)
            except (json.JSONDecodeError, ValueError):
                continue

            if not pts or len(pts) != 21:
                status = "no_hand"
                letter = None
            else:
                # تتبع السبابة والخنصر للـ J و Z
                j_z_history.append((pts[8], pts[20]))

                # نعمل snapshot من j_z_history عشان Thread Safety
                history_snapshot = list(j_z_history)

                # التوقع
                letter, status = await loop.run_in_executor(None, predict_from_pytorch, pts, history_snapshot)

            smoothed_letter = None
            confirmed_letter = None

            if letter and status == "success":
                history.append(letter)
                smoothed_letter = Counter(history).most_common(1)[0][0]

                if smoothed_letter == last_confirmed_letter:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                    last_confirmed_letter = smoothed_letter

                # لو الحرف ثبت لـ 5 إطارات متتالية
                if consecutive_count == REQUIRED_CONSECUTIVE:
                    confirmed_letter = smoothed_letter
                    if confirmed_letter == "Space": # لو ضفت لوجيك للمسافة
                        current_sentence += (" " + current_word) if current_sentence else current_word
                        current_word = ""
                    else:
                        current_word += confirmed_letter
                    
                    history.clear()
                    consecutive_count = 0

            # اقتراح الكلمات — بيتحسب بس لما الكلمة تتغير (cache)
            if current_word and len(current_word) >= 2:
                if current_word != last_word_for_suggestions:
                    cached_suggestions = get_suggestions(current_word)
                    last_word_for_suggestions = current_word
                suggestions = cached_suggestions
            else:
                suggestions = []
                last_word_for_suggestions = ""

            display_text = current_sentence
            if current_word:
                display_text += (" " + current_word) if display_text else current_word

            # إرسال النتيجة للموبايل
            await websocket.send_json({
                "status": status,
                "raw_prediction": letter,
                "confirmed_letter": confirmed_letter,
                "current_word": current_word,
                "final_word": display_text,
                "suggestions": suggestions
            })

    except WebSocketDisconnect:
        print("Flutter Client Disconnected")
    except Exception as e:
        print(f"Error: {e}")