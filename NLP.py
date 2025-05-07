import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt_tab")  # ใช้ชื่อ 'punkt_tab' ตามที่ระบบแจ้ง
# nltk.download("popular")  # ดาวน์โหลดข้อมูลพื้นฐานทั้งหมด

df = pd.read_excel(
    rf"D:\OneFile\WorkOnly\AllCode\Python\NLP\ICD-10-TM-Rath-master\ICD10TM-Public.xlsx"
)
df.dropna()
df = df[["Code", "Term"]]
print(df)


def custom_tokenizer(text):
    text = text.lower()  # ตัวพิมพ์เล็ก
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # ลบเครื่องหมายวรรคตอน
    tokens = word_tokenize(text)  # ตัดคำ
    stop_words = set(stopwords.words("english"))  # คำหยุด
    filtered_tokens = [word for word in tokens if word not in stop_words]  # กรองคำหยุด
    return filtered_tokens


# แปลงรหัสโรคเป็นตัวเลข
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Code"])

# แบ่งข้อมูลฝึกอบรมและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(
    df["Term"], y, test_size=0.2, random_state=42
)

# สร้าง pipeline การประมวลผล
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=custom_tokenizer,
                lowercase=False,
                max_features=5000,  # จำกัดจำนวน features
                ngram_range=(1, 2),  # ใช้ unigram + bigram
            ),
        ),
        (
            "lgr",
            LogisticRegression(
                max_iter=1000,
                solver="saga",  # ใช้ solver ที่เหมาะกับข้อมูลขนาดใหญ่
                penalty="l2",
                C=0.5,
            ),
        ),
    ]
)

print("Training...")
# ฝึกโมเดล
pipeline.fit(X_train, y_train)

print("Evaluating...")
# ประเมินประสิทธิภาพ
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# ฟังก์ชันทำนายรหัสโรค
def predict_icd10(text):
    predicted = pipeline.predict([text])
    return label_encoder.inverse_transform(predicted)[0]


# ทดลองใช้
test_text = "Patient with severe asthma attack"
predicted_code = predict_icd10(test_text)
print(f"Input Text: {test_text}")
print(f"Predicted ICD-10 Code: {predicted_code}")
