# Andhra Pradesh crop-yield-prediction
# 🌾AP Crop Yield Prediction – A Machine Learning Based Web App

This project predicts **agricultural crop yield** of the **Andhra Pradesh** using machine learning models based on various input features like district, crop, season, and area. The model is trained on real-world datasets, focusing on data from **Andhra Pradesh**.

### 🔗 Live Demo  
👉 [Check the Web App Here](https://guna00-crop-yield-prediction.hf.space)

---

## 📌 Features

- User-friendly **Streamlit** interface  
- Predicts crop yield based on inputs  
- Uses **machine learning models** like Linear Regression, Random Forest, etc.  
- Handles preprocessing with **feature scaling and encoding**  
- Supports **real-time predictions**  

---

## 🚀 Tech Stack

- **Python**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Streamlit** for frontend
- **Render** for deployment

---

## 🧠 Machine Learning

- Models used: `Linear Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`
- Evaluation Metrics: R² Score, MAE, RMSE
- Final model selected based on highest performance and accuracy

---

## 📁 Project Structure

```
crop-yield-prediction/
│
├── website.py              # Streamlit web app script
├── yield_model.pkl         # Trained ML model
├── scaler.pkl              # StandardScaler for preprocessing
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔧 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/Gunasekhar00212/crop-yield-prediction.git
   cd crop-yield-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:
   ```bash
   streamlit run website.py
   ```

---

## 📊 Sample Inputs
 
- **District**: Krishna  
- **Crop**: Rice  
- **Season**: Kharif  
- **Area**: 1000  


---

## 📬 Contact

Made with ❤️ by **Ande Guna Sekhar**  
📧 andegunashekar@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/guna-sekhar-ande-a3094523b)
