# 1. نستخدم نسخة بايثون خفيفة ومستقرة
FROM python:3.10-slim

# 2. تحديث قائمة الباكدجات بدون تسطيب مكتبات غير مستخدمة
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# 3. تحديد مجلد العمل جوه الكونتينر
WORKDIR /app

# 4. نسخ ملف المتطلبات الأول (عشان الدوكر يعمل Caching ويسرع البناء بعدين)
COPY requirements.txt .

# 5. تسطيب مكتبات البايثون
# --extra-index-url عشان pip يجيب torch بنسخة CPU فقط (بدون CUDA ~2GB)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# 6. نسخ باقي ملفات المشروع (app.py, train_model.py, sign_language_model.pth)
COPY . .

# 7. فتح البورت اللي FastAPI بيشتغل عليه
EXPOSE 8000

# 8. أمر تشغيل السيرفر لما الكونتينر يشتغل
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]