import matplotlib.pyplot as plt
from pandas import read_csv
import os
import numpy as np

# Load data
data_path = os.path.join(os.getcwd(), "data\\blood-pressure.txt")
dataset = read_csv(data_path, delim_whitespace=True)

# We have 30 entries in our dataset and four features. The first feature is the ID of the entry.
# The second feature is always 1. The third feature is the age and the last feature is the blood pressure.
# We will now drop the ID and One feature for now, as this is not important.
dataset = dataset.drop(['ID', 'One'], axis=1)

# And we will display this graph
#%matplotlib inline
plt.scatter(x=dataset["Age"], y=dataset["Pressure"])

# Now, we will assume that we already know the hypothesis and it looks like a straight line
h = lambda x: 84 + 1.24 * x

# Let's add this line on the chart now
ages = range(18, 85)
estimated = []

for i in ages:
    estimated.append(h(i))

plt.plot(ages, estimated, 'b')  
plt.show()


# Let's calculate the cost for the hypothesis above

hypothesis = lambda x, theta_0, theta_1: theta_0 + theta_1 * x

def cost(X, y, t0, t1):
    m = len(X) # the number of the training samples
    c = np.power(np.subtract(hypothesis(X, t0, t1), y), 2)
    print(sum(c))
    return (1 / (2 * m)) * sum(c)

X = dataset.values[:, 0]
y = dataset.values[:, 1]
print('J(Theta) = %2.2f' % cost(X, y, 84, 1.24))