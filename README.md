
# 🛫 Flight Delay Prediction Using Machine Learning

### 🚀 Large-Scale Supervised Learning Project | Data Science | Predictive Analytics

---

## 📘 Overview

Air travel delay prediction is a critical challenge in the aviation industry.  
This project leverages **Machine Learning** on the **U.S. Department of Transportation (DOT) Flight Delay Dataset** (with millions of records) to predict whether a flight will arrive on time or be delayed.

The project demonstrates advanced **data preprocessing**, **feature engineering**, and **model optimization** — essential skills for data scientists working with real-world, high-volume data.

---

## 🧠 Objectives

- Analyze millions of flight records to identify delay patterns.  
- Build a predictive model to classify **On-Time (0)** or **Delayed (1)** flights.  
- Handle **large-scale datasets efficiently** using optimized Pandas/Numpy workflows.  
- Apply **supervised learning algorithms** for classification and regression.  
- Deliver actionable **business insights** for airlines and airport authorities.

---

## 📊 Dataset Details

- **Source:** [US DOT - Airline Delay and Cancellation Data (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays)  
- **Size:** ~5.8 million rows  
- **Period:** 2015 – 2019  
- **Key Columns:**  
  - `FL_DATE`: Flight Date  
  - `OP_CARRIER`: Airline Code  
  - `ORIGIN`, `DEST`: Airports  
  - `DEP_DELAY`, `ARR_DELAY`: Departure/Arrival Delays (in minutes)  
  - `DISTANCE`: Distance between airports  
  - `CANCELLED`, `DIVERTED`: Binary flags  

- **Target Variable:**  
  - `DELAYED` = 1 if `ARR_DELAY` > 15 minutes  
  - `DELAYED` = 0 otherwise

---

## ⚙️ Tech Stack

| Category | Tools Used |
|-----------|------------|
| Language | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Optimization | GridSearchCV |
| Model Saving | Joblib |
| Notebook | Jupyter Notebook |
| Optional Deployment | Streamlit / Flask |

---

## 🧩 Project Pipeline

### **1️⃣ Data Loading & Cleaning**
- Load multi-million-row CSV using `chunksize` for efficiency.  
- Drop irrelevant features like `TAIL_NUM`, `FLIGHT_NUM`.  
- Handle missing values (`NaN` in delay columns).  
- Create target column `DELAYED`.

### **2️⃣ Feature Engineering**
- Extract time-based features:
  - `Month`, `Day_of_Week`, `Hour_of_Day`
- Encode categorical columns (`Airline`, `Origin`, `Dest`).
- Normalize distance using `StandardScaler`.

### **3️⃣ Exploratory Data Analysis (EDA)**
- Analyze **delay distribution by airline, day, and airport**.  
- Identify **seasonal trends** and **peak delay hours**.  
- Visualize delay patterns using heatmaps and bar charts.

### **4️⃣ Model Building**
- Train/Test Split (80/20)
- Models Used:
  - Logistic Regression (Baseline)
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost (Final)

### **5️⃣ Model Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Visuals: Confusion Matrix, ROC Curve, Feature Importance  
- Feature importance interpretation using `SHAP` and `LIME`

### **6️⃣ Model Optimization**
- Grid Search for hyperparameters:
  - `n_estimators`, `max_depth`, `learning_rate`
- Early stopping for XGBoost to prevent overfitting

### **7️⃣ Model Saving**
- Save best-performing model as `flight_delay_model.pkl`

---

## 📈 Results

| Model | Accuracy | ROC-AUC | F1 Score |
|--------|-----------|----------|-----------|
| Logistic Regression | 0.77 | 0.80 | 0.75 |
| Random Forest | 0.84 | 0.88 | 0.83 |
| **XGBoost (Best)** | **0.87** | **0.90** | **0.86** |

✅ The XGBoost model performed best on unseen test data.  
✅ High ROC-AUC indicates strong classification performance.  
✅ Excellent scalability on multi-million-row data.

---

## 💡 Key Business Insights

- **Peak delays** occur between **3 PM – 9 PM** (evening flights).  
- Airlines with hub congestion show higher average delay times.  
- Winter months (Dec–Feb) and bad-weather regions have more delays.  
- **Feature Importance:**  
  - Airline  
  - Departure Hour  
  - Day of Week  
  - Distance  

> ✈️ *Airlines can use this model to forecast delays and optimize scheduling to improve on-time performance.*

---

## 🧰 Folder Structure

```

Flight_Delay_Prediction_Using_Machine_Learning/
│
├── data/
│   └── flights_large_dataset.csv
│
├── notebooks/
│   └── Flight_Delay_Prediction.ipynb
│
├── model/
│   └── flight_delay_model.pkl
│
├── images/
│   ├── feature_importance.png
│   ├── delay_by_airline.png
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md

```

---

## 🧑‍💻 Author

**👋 Avinash Kamble**  
🎓 IT Engineering Student | Aspiring Data Scientist  
📍 Mumbai, India  
💼 Focus Areas: Machine Learning | AI | Data Engineering  
🌐 [GitHub](https://github.com/avinash-kamble-9) • [LinkedIn](https://linkedin.com/in/avinashzz)

---

## ⭐ How to Support

If you find this project useful:
- Give it a ⭐ on GitHub  
- Share it with peers  
- Connect on LinkedIn for collaborations 🤝  

---

> “Predicting delays is easy — preventing them is data science.” ✈️  
> — *Avinash Kamble*
```


---

### ✅ **Folder Structure**

```
Flight_Delay_Prediction_Using_Machine_Learning/
│
├── data/
│   └── flight_delay_dataset.csv               # Kaggle dataset file
│
├── notebooks/
│   └── Flight_Delay_Prediction.ipynb  # notebook 
│
├── model/
│   └── flight_delay_model.pkl
│
├── images/
│   ├── delay_distribution.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
└── README.md
```


