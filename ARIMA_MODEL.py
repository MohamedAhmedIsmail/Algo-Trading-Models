import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 

series =pd.read_csv("C:\\Users\\mohamed ismail\\Desktop\\GP And Data\\Telecom Egypt.csv")
X = series.iloc[:,-1].values

size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]

history = [x for x in train]
predictions = list()

for t in range(len(test)):#
	model = ARIMA(history, order=(6,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test,color='blue',label="actual")
pyplot.plot(predictions, color='red',label="prediction")
pyplot.legend()
pyplot.show()