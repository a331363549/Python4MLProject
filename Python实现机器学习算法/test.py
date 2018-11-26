import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

data = pd.read_csv('cancer.csv',header=0)
print(data.describe())   # 根据这个统计，我们可对下列的坐标 x,y 进行相应的调整
# 原始数据
plt.figure()      # Create a new figure
# 坐标轴 x 从 0 ~ 70，y 从 -0.2 ~ 1.2
plt.axis([0,70,-0.2,1.2])
plt.title('Original data')
plt.scatter(data['mean radius'],data['type'])

# 数据中心零均值处理
plt.figure()
plt.axis([-30,30,-0.2,1.2])
plt.title('Zero mean')
plt.scatter(data['mean radius'] - 14.127292,data['type'])
plt.show()

plt.figure()
plt.axis([-5,5,-0.2,1.2])
plt.title('Scaled by std dev')
plt.scatter((data['mean radius']-14.127292)/3.52,data['type'])
plt.show()
print('\n',(data['mean radius']/3.524049).describe())
