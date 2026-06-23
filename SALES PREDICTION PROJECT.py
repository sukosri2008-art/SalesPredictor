import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv(r"C:\Users\srija\Downloads\Advertising Budget and Sales.csv")
print("FIRST 5 ROWS :")
print(data.head(5))
X= data.drop("Sales ($)",axis=1)
y=data["Sales ($)"]
model=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print("Real values",y_test)
accuracy=r2_score(y_test,prediction)
print("ACCURACY SCORE: ",round (accuracy*100))
for i in range(10):
    print("ACTUAL",y_test.iloc[i],
          "||  PREDICTED ",prediction[i])
