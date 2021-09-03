import pandas as pd
import csv
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
slept = df["Hours_Slept"].tolist()
studied = df["Hours_studied"].tolist()

colours = []
hours = df[["Hours_studied", "Hours_Slept"]]
results = df["results"]

hours_train, hours_test, results_train, results_test = train_test_split(hours, results, test_size = 0.25, random_state = 0)
classifier = LogisticRegression(random_state = 0)
classifier.fit(hours_train, results_train)
LogisticRegression(C = 1.0, class_weight = None, dual = False, fit_intercept = True, intercept_scaling = 1, l1_ratio = None, max_iter = 100, multi_class = "auto", n_jobs = None, penalty = "l2", random_state = 0, solver = "lbfgs", tol = 0.0001, verbose = 0, warm_start = False)

#test the accuracy of the model
resultspred = classifier.predict(hours_test)
print("Accuracy: ", accuracy_score(results_test, resultspred))

userhourstudied = int(input("Enter the hours studied: "))
userhourslept = int(input("Enter the hours slept: "))
sc_x = StandardScaler()
usertest = sc_x.transform([[userhourstudied, userhourslept]])
userresultprediction = classifier.predict(usertest)
if userresultprediction[0] == 1:
    print("The user may pass")
else:
    print("The user may not pass")

for data in results:
    if data==1:
        colours.append("green")
    else:
        colours.append("red")

fig = go.Figure(data = go.Scatter(
    x = studied,
    y = slept,
    mode = "markers",
    marker = dict(color = colours)
))
fig.show()