import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

twitter_data = pd.read_csv('result.csv')

plt.figure()
hist1,edges1 = np.histogram(twitter_data.friends)
plt.bar(edges1[:-1],hist1,width=edges1[1:]-edges1[:-1])

plt.scatter(twitter_data.followers,twitter_data.retwc)
plt.scatter(twitter_data.followers,twitter_data.friends)
y = twitter_data.friends
X = twitter_data.followers
X = sm.add_constant(X)
lr_model = sm.OLS(y,X).fit()
print(lr_model.summary())
X_prime = np.linspace(X.followers.min(),X.followers.max(),100)
X_prime = sm.add_constant(X_prime)
y_hat = lr_model.predict(X_prime)
plt.scatter(X.followers,y)
plt.xlabel("followers")
plt.ylabel("friends")
plt.plot(X_prime[:,1],y_hat)