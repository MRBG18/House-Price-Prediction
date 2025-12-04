# 🏡 Ahmedabad House Price Prediction (Machine Learning + Tkinter GUI)

A ML-powered desktop application that predicts **house prices in Ahmedabad** based on key real-estate features.  
The system automatically performs **data cleaning, feature engineering, outlier handling**, trains a **Linear Regression model**, and provides **real-time price predictions through a Tkinter GUI**.

---

## 🚀 Project Highlights

| Feature | Description |
|--------|-------------|
| 🔹 Fully Automated ML Pipeline | Cleans data, handles missing values, removes outliers |
| 🔹 Feature Selection (SFS) | Selects **Top 7 most influential features** |
| 🔹 Linear Regression Model | Simple, fast & explainable prediction |
| 🔹 GUI Desktop Application | User inputs property details → Instant price prediction |
| 🔹 Uses ML + Real-Estate Domain Approach | Suited for real-time buyer/seller use |
| 🔹 Results Shown in ₹ | Indian currency formatted dynamically |

---

## 📊 Machine Learning Workflow

1️⃣ Load dataset (`dataset.csv`)  
2️⃣ Handle missing numeric values using **mean imputation**  
3️⃣ Detect & remove outliers using **IQR Filtering**  
4️⃣ Select best features using **Sequential Feature Selector (mlxtend)**  
5️⃣ Train & test ML model using **Linear Regression**  
6️⃣ Evaluate performance → Print accuracy  
7️⃣ Deploy final model using Tkinter GUI  

---

## 🔍 Feature Selection Code (Core Step)

```python
lr = LinearRegression()
fs = SequentialFeatureSelector(lr, k_features=7, forward=True)
fs.fit(x, y)

print("Selected Features:", fs.k_feature_names_)
print("Feature Score:", fs.k_score_ * 100)
