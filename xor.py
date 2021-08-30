# Logistic Regression dùng thư viện sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Load data từ file 
x=np.array([[0,0],[1,1],[0,1],[1,0]]).reshape(-1,2)
y=np.array([0,0,1,1]).reshape(-1,1)
# Vẽ data bằng scatter
plt.scatter(x[:2, 0], x[:2, 1], c='red', edgecolors='none', s=30, label="output=0")
plt.scatter(x[2:, 0], x[2:, 1], c='blue', edgecolors='none', s=30, label='output=1')
plt.legend(loc=1)
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')
# Tạo mô hình Logistic Regression và train
logreg = LogisticRegression()
logreg.fit(x, y)
# Lưu các biến của mô hình vào mảng
wg = np.zeros( (3, 1) )
wg[0, 0] = logreg.intercept_
wg[1:, 0] = logreg.coef_
# Vẽ đường phân cách
t = 0.5
plt.plot((0, 0.5),(-(wg[0]+0*wg[1]+ np.log(1/t-1))/wg[2], -(wg[0] + 0.5*wg[1]+ np.log(1/t-1))/wg[2]), 'g')
plt.show()
# Lưu các tham số dùng numpy.savez(), đỉnh dạng '.npz'

np.savez('w_logistic2.npz', a=logreg.intercept_, b=logreg.coef_)
# Load các tham số dùng numpy.load(), file '.npz'
k = np.load('w_logistic2.npz')
logreg.intercept_ = k['a']
logreg.coef_ = k['b']