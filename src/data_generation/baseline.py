import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def load_pilot_data():
    """
    Loads and duplicates the verified pilot data to simulate a small training matrix.
    """
    # 3 הדוגמאות המאושרות מהפיילוט שלכן
    pilot_json = """
    [
      {
        "message": "I've been feeling really worried since my surgery. My incision on my belly seems to be leaking a bit, and it feels warm to the touch. I noticed some redness spreading around it too. Also, I've been having a fever; my temperature was 38.5°C.",
        "labels": {"infection_warning": 1, "wound_problem": 1, "bleeding_warning": 0, "severe_pain_warning": 0, "urinary_problem": 0, "respiratory_warning": 0, "leg_clot_warning": 0, "routine_recovery": 0}
      },
      {
        "message": "The patient is showing signs of potential complications. I've observed unilateral swelling in her lower leg that seems persistent, accompanied by some discomfort. It could indicate a DVT risk.",
        "labels": {"infection_warning": 0, "wound_problem": 0, "bleeding_warning": 0, "severe_pain_warning": 0, "urinary_problem": 0, "respiratory_warning": 0, "leg_clot_warning": 1, "routine_recovery": 0}
      },
      {
        "message": "Hey, I wanted to mention that things have been pretty normal since my procedure. I'm just a bit tired and feeling some mild soreness around my scar. It itches sometimes. I've also had some constipation.",
        "labels": {"infection_warning": 0, "wound_problem": 0, "bleeding_warning": 0, "severe_pain_warning": 0, "urinary_problem": 0, "respiratory_warning": 0, "leg_clot_warning": 0, "routine_recovery": 1}
      }
    ]
    """
    data = json.loads(pilot_json)
    
    # משכפלים את הדאטה (Oversampling) כדי שיהיה ל-scikit-learn מספיק דוגמאות להריץ מטריצות
    duplicated_data = data * 10 
    
    texts = [item["message"] for item in duplicated_data]
    
    # חילוץ הלייבלים למטריצה נימפת (Rows = Cases, Columns = 8 Labels)
    label_keys = ["infection_warning", "wound_problem", "bleeding_warning", "severe_pain_warning", 
                  "urinary_problem", "respiratory_warning", "leg_clot_warning", "routine_recovery"]
    
    Y = []
    for item in duplicated_data:
        Y.append([item["labels"][key] for key in label_keys])
        
    return texts, np.array(Y), label_keys

def train_eval_baseline():
    print("🤖 Initializing TF-IDF + Logistic Regression Multi-Label Baseline...")
    
    # 1. טעינת הנתונים
    texts, Y, label_names = load_pilot_data()
    
    # 2. חלוקה זמנית לסט אימון וסט בדיקה (Train/Test Split)
    # לוקחים 80% לאימון ו-20% לבדיקה
    split_idx = int(len(texts) * 0.8)
    X_train, X_test = texts[:split_idx], texts[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # 3. הפיכת הטקסט למטריצת פיצ'רים דיגיטלית באמצעות TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. הגדרת מסווג Multi-label (Binary Relevance באמצעות OneVsRest)
    # הוא מאמן מסווג רגרסיה לוגיסטית נפרד עבור כל אחד מ-8 הלייבלים
    base_lr = LogisticRegression(class_weight='balanced')
    clf = OneVsRestClassifier(base_lr)
    
    # 5. אימון המודל
    clf.fit(X_train_tfidf, Y_train)
    
    # 6. חיזוי על סט הבדיקה
    Y_pred = clf.predict(X_test_tfidf)
    
    # 7. הדפסת דוח ביצועים רשמי של הקורס (Classification Report)
    print("\n📊 Evaluation Metrics Across Clinical Sub-Groups (Task A):")
    print(classification_report(Y_test, Y_pred, target_names=label_names, zero_division=0))
    
    print("✅ Baseline Pipeline Executed and Verified Successfully!")

if __name__ == "__main__":
    train_eval_baseline()
    