# 🥑 Avocado Price Prediction Using Machine Learning

An end-to-end machine learning project to predict avocado prices using historical data and regression models. This project helps businesses, retailers, and suppliers understand price trends, regional differences, and seasonal effects for better decision-making.

An interactive **Streamlit** web app is included to visualize insights and make real-time price predictions.

---

## 📌 Project Overview

This project uses historical avocado data to build and evaluate multiple regression models for price prediction. It covers data preprocessing, exploratory data analysis (EDA), model training, model comparison, and deployment as a Streamlit web application.

---

## 🎯 Problem Statement

Avocado prices change due to multiple factors such as:

- Region  
- Demand  
- Seasonality  
- Type  

Accurately predicting prices helps to:

- Optimize pricing strategies  
- Improve supply chain planning  
- Reduce losses caused by price volatility  

The goal is to build a complete machine learning pipeline that can efficiently predict avocado prices.

---

## 🧠 Project Workflow

1. **Data Collection**
   - Historical avocado dataset (`avocado.csv`)

2. **Data Cleaning & Preprocessing**
   - Handling missing values  
   - Encoding categorical variables  
   - Feature selection  

3. **Exploratory Data Analysis (EDA)**
   - Price trends over time  
   - Region-wise price comparison  
   - Seasonal patterns and behavior  

4. **Model Training & Evaluation**
   - Training multiple regression models  
   - Evaluating models using metrics like R² Score, Mean Squared Error (MSE), and Mean Absolute Error (MAE)  

5. **Model Comparison**
   - Comparing model performance to identify the best-performing model  

6. **Deployment**
   - Building an interactive **Streamlit** application for prediction and visualization  

---

## 🛠 Tech Stack

- **Programming Language:** Python  

- **Libraries:**
  - Pandas  
  - NumPy  
  - Matplotlib  
  - Seaborn  
  - Scikit-learn  

- **Web Framework:** Streamlit  

---

## 📊 Machine Learning Models

The following regression models are implemented and evaluated:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Ridge Regression  
- Lasso Regression  

---

## 🏆 Best Performing Model

The models are evaluated using:

- R² Score  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  

✅ **Random Forest Regressor** achieved the best overall performance and is selected as the final model in the application.

---

## 📈 Key Insights

- Avocado prices vary significantly across regions  
- Seasonality has a strong impact on pricing patterns  
- Machine learning models can effectively capture complex price behavior  
- Ensemble models such as Random Forest perform better than simple linear models for this prediction task  

---

## 🚀 Streamlit Application

The Streamlit app allows users to:

- Explore EDA visualizations  
- Compare regression model performance  
- Predict avocado prices interactively based on input features  

---

## ▶️ How to Run Locally

# Clone the repository
```
git clone https://github.com/AnjaliPanduga/Avocado-Price-Prediction.git
cd Avocado-Price-Prediction
```
# Install dependencies
```
pip install -r requirements.txt
```
# Run the Streamlit app
```
streamlit run app.py
```
---
# 📂 Project Structure

```
├── EDA Of Avocado Dataset.ipynb
├── Price Regression.ipynb
├── Comparision of all regression models.ipynb
├── app.py
├── avocado.csv
├── requirements.txt
└── README.md
```
---
# 📌 Business Value

This project shows how data-driven approaches can help businesses:

- Forecast avocado prices more accurately
- Understand demand and regional behavior
- Make informed and smarter pricing decisions
---

# 🙌 Conclusion

This project covers the complete machine learning lifecycle:

- Data analysis
- Feature engineering
- Regression modeling
- Model comparison
- Deployment with Streamlit

It demonstrates practical skills in EDA, regression, model evaluation, and building interactive ML applications suitable for real-world business scenarios.

