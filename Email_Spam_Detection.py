# Email_Spam_Detection.py
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


model = MultinomialNB()
cv = CountVectorizer()


# Load Dataset
data = pd.read_csv("emails.csv")

emails = data["text"]


# Preprocess
x = cv.fit_transform(emails)
y = data["spam"]

# Train
model.fit(x, y)

# Prediction
test_email = "hello bg , you are winning 1 lack rupees follow link and just collect you money"
test_vector = cv.transform([test_email])
prediction = result = model.predict(test_vector)

Accucary = model.score(x,y)
print("Model Accuracy :- ",Accucary*100,"%")

#cheaking input length
if len(test_email.split()) < 5:
    print("Message too short to classify reliably.")
else:
    if prediction == 1:
        print('Mail: ', test_email)
        print("Received Mail Is Spam")
    else:
        print('Mail', test_email)
        print("Received Mail Is Not Spam")
