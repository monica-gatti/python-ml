from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pandas import read_csv
import os

# LinearRegression uses the gradient descent method

data_path = os.path.join(os.getcwd(), "data\\blood-pressure.txt")
dataset = read_csv(data_path, delim_whitespace=True)

# We have 30 entries in our dataset and four features. The first feature is the ID of the entry.
# The second feature is always 1. The third feature is the age and the last feature is the blood pressure.
# We will now drop the ID and One feature for now, as this is not important.
dataset = dataset.drop(['ID', 'One'], axis=1)
# Our data
X = dataset[['Age']]
y = dataset[['Pressure']]

regr = LinearRegression()
regr.fit(X, y)

# Plot outputs
plt.xlabel('Age')
plt.ylabel('Blood pressure')

plt.scatter(X, y,  color='black')
plt.plot(X, regr.predict(X), color='blue')

plt.show()
plt.gcf().clear()



print( 'Predicted blood pressure at 25 y.o.   = ', regr.predict(25) )
print( 'Predicted blood pressure at 45 y.o.   = ', regr.predict(45) )
print( 'Predicted blood pressure at 27 y.o.   = ', regr.predict(27) )
print( 'Predicted blood pressure at 34.5 y.o. = ', regr.predict(34.5) )
print( 'Predicted blood pressure at 78 y.o.   = ', regr.predict(78) )

