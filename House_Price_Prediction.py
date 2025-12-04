import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
import tkinter as tk
from tkinter import ttk, messagebox

data=pd.read_csv('dataset.csv')

'''
print(train_data.head())
print(test_data.head())

'''
#Shape of dataFrames
print("\nTrain Data Shape:", data.shape)

#Info about Data
#print(data.info())


#Print and check Null Values
print("\n\n<--- Missing Values Count per Column --->")
print(data.isna().sum())

for col in data.select_dtypes(include='number'):
    data[col] = data[col].fillna(data[col].mean())

#outlier detection
"""
for col in data.select_dtypes(include='number'):
    sns.boxplot(x=col,data=data)
    plt.show()
"""

#Columns With Outlier
#srft lot_size price

outliercol = ['sqft', 'lot_size', 'price']

for col in outliercol:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1

    min_r = q1 - (1.5 * iqr)
    max_r = q3 + (1.5 * iqr)

    data = data.loc[(data[col] >= min_r) & (data[col] <= max_r)]

'''
for col in data.select_dtypes(include='number'):
    sns.boxplot(x=col,data=data)
    plt.show()
'''

print('\n<--- After Data Cleaning --->')
print(data.isna().sum())

x = data.iloc[:, :-1]
y = data['price']

lr = LinearRegression()

fs = SequentialFeatureSelector(lr,k_features=7,forward=True)
fs.fit(x,y)
print("Best Columns: ",fs.k_feature_names_)
print("Features Score: ",fs.k_score_*100)


#Feature_List = ['sqft', 'bedrooms', 'bathrooms', 'age', 'garage', 'distance_city', 'school_rating']
Feature_List =list(fs.k_feature_names_)

#Corelation
#sns.heatmap(data=data[Feature_List].corr(),annot=True)
#plt.show()

x = data[Feature_List]
y = data['price']

# Train and Test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Model Training
lr.fit(x_train,y_train)

print("Model Accuracy: ",lr.score(x_test,y_test)*100)

"""
sqrf = float(input("Enter SQFT: "))
bedrooms = float(input("Enter Bedrooms: "))
bathrooms = float(input("Enter Bathrooms: "))
age = float(input("Enter Age: "))
garage = float(input("Enter Garages: "))
distance_city = float(input("Enter Distance_City: "))
school_rating = float(input("Enter School: "))

result =  lr.predict([[sqrf,bedrooms,bathrooms,age,garage,distance_city,school_rating]])
print("Predicted Price : ",result)"""



#===================
#     GUI PART
#==================


# Predict button callback
def on_predict():
    try:
        sqrf = float(entry_sqft.get())
        bedrooms = float(entry_bedrooms.get())
        bathrooms = float(entry_bathrooms.get())
        age = float(entry_age.get())
        garage = float(entry_garage.get())
        distance_city = float(entry_distance.get())
        school_rating = float(entry_school.get())

        input_df = pd.DataFrame([[sqrf, bedrooms, bathrooms, age, garage, distance_city, school_rating]], columns=Feature_List)
        result = lr.predict(input_df)

        messagebox.showinfo("Predicted Price","Predicted Price:- "+"₹"+str(int(np.round(result).item())))

    except ValueError:
        messagebox.showerror("Error","Please Enter Valid Data")





root = tk.Tk()
root.title(" Ahmedabad House Price Predictor Model")
root.geometry("450x500")
root.resizable(False, False)

style = ttk.Style()
style.configure("TLabel", font=("Arial", 12))
style.configure("TButton", font=("Arial", 13, "bold"))
style.configure("TEntry", font=("Arial", 12))

# INPUT FIELDS
tt_label = ttk.Label(root, text="Enter Property Details", font=("Arial", 18, "bold"))
tt_label.pack(pady=15)

frame = tk.Frame(root)
frame.pack(pady=10)

def add_input(label_text):
    row = tk.Frame(frame)
    row.pack(fill="x", pady=5)
    lbl = ttk.Label(row, text=label_text, width=18)
    lbl.pack(side="left")
    entry = ttk.Entry(row)
    entry.pack(side="right", expand=True, fill="x")
    return entry

entry_sqft = add_input("Area (sqft):")
entry_bedrooms = add_input("Bedrooms:")
entry_bathrooms = add_input("Bathrooms:")
entry_age = add_input("Age (Years):")
entry_garage = add_input("Garage Space:")
entry_distance = add_input("Distance to City:")
entry_school = add_input("School Rating:")

predict_btn = ttk.Button(root, text="Predict Price",command=on_predict)
predict_btn.pack(pady=18)



root.mainloop()








