import numpy as np
import pandas as pd
import matplotlib,pyplot as plt

height = np.array([183, 150, 180, 197, 160, 175])
height = height.reshape(-1, 1) # 2차원 값이 들아가야 하기에 reshape를 사용한다.

math = np.array([85, 45, 80, 99, 45, 75])

from sklearn.linear_model import LinearRegression   #scikit-learn(sklearn)에 선형회귀 라이브러리가 포함되어 있어서 사용한다.

line_fitter = LinearRegression()

# fit()함수의 기능 
# line_fitter.coef_ : 기울기를 저장한다. (coef : 기울기)
# line_fitter.intercept_ : 절편을 저장 (intercept : 절편)

line_fitter.fit(height, math) #height와 math관의 상관관계를 구한다.

score_predict = line_fitter.predict(height) ##height와 math관의 상관관계를 구하는 선형회귀선을 그린다. (score_predict : 점수 예측)

plt.plot(height, math, 'x') #x는 height에 따른 math이다. (그래프에 나타내는 표시를 x로 한다.)
plt.plot(height, score_predict)
plt.show() #데이터들을 나타낸 선형회귀선과 height에 따른 math을 나타내는 x를 출력한다. 

#위에서 구현한 그래프에 기울기와 절편을 표현해보자.

line_fitter.coef_ #기울기 구하기

line_fitter.intercept_ #절편 구하기

#MSE를 이용해서 성능평가 진행 (성능평가를 위해서 mse를 import한다.)
from sklearn.metrics import mean_squared_error

print("Mean_Squared_Error(Mse) :", mean_squared_error(score_predict, math))#mse값을 구한다.

#RMSE를 이용해서 성능평가 진행(RMSE는 MSE에 루트값을 씌우면 된다.)
print("RMSE : ", mean_squared_error(score_predict, math)**0.5) # **0.5 : 루트를 씌운다.

# LinearRegression의 score를 구한다.
print('score :', line_fitter.score(height, math)) #LinearRegression을 line_fitter로 정의하였기 때문에 다음과 같이 사용한다.