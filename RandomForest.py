import csv,pandas,re
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 0 读取数据
oridata = []
with open('a.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        oridata.append(row)
# print(oridata)

# 1 处理数据
data = []
Street = {}
Community = {}
Heading = {'East':'1','South':'1.1','West':'1.2','North':'1.3','east':'1','south':'1.1','west':'1.2','north':'1.3'}
countStreet = 0
countCommunity = 0
countHeading = 0
for i in range(len(oridata)-1):
    item = oridata[i+1]
    line = ['']*16
    data.append(line)

    if item[1] in Street.keys():
        data[i][0] = Street[item[1]]
    else:
        data[i][0] = countStreet
        Street[item[1]] = countStreet
        countStreet = countStreet + 1
    
    if item[2] in Community.keys():
        data[i][1] = Community[item[2]]
    else:
        data[i][1] = countCommunity
        Community[item[2]] = countCommunity
        countCommunity = countCommunity + 1
    
    headscore = 0
    for head in Heading.keys():
        if head in item[9]:
            headscore = headscore + float(Heading[head])
    if headscore > 2:
        headscore = headscore /2
    data[i][8] = int((headscore - 1)*20)


    data[i][11] = re.findall('(.*?)㎡',(item[12][:-1]),re.S)[0]

    if 'Yes' in item[14]:
        data[i][13] = 1
    else:
        data[i][13] = 0


    if 'Fine' in item[16]:
        data[i][15] = 1
    else:
        data[i][15] = 0

    if 'High' in  item[15]:
        data[i][14] = 2
    elif 'Mid' in item[15]:
        data[i][14] = 1
    else:
        data[i][14] = 0

    for j in [3,4,5,6,7,8,10,11,13]:
        data[i][j-1] = float(item[j])

xx = []
yy = []
for i in range(len(data)):
    yy.append(data[i][2])
    xx.append([])
    for j in [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        xx[i].append(data[i][j])
x = np.array(xx)
y = np.array(yy)
# print(len(x),len(y))

# 2 分割训练数据和测试数据
# 随机采样25%作为测试 75%作为训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# 3 训练数据和测试数据进行标准化处理
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 4 三种集成回归模型进行训练和预测
# 随机森林回归
rfr = RandomForestRegressor()
# 训练
rfr.fit(x_train, y_train)
# 预测 保存预测结果
rfr_y_predict = rfr.predict(x_test)

# 极端随机森林回归
etr = ExtraTreesRegressor()
# 训练
etr.fit(x_train, y_train)
# 预测 保存预测结果
etr_y_predict = rfr.predict(x_test)

# 梯度提升回归
gbr = GradientBoostingRegressor()
# 训练
gbr.fit(x_train, y_train)
# 预测 保存预测结果
gbr_y_predict = rfr.predict(x_test)

# 5 模型评估
# 随机森林回归模型评估
print("随机森林回归的默认评估值为：", rfr.score(x_test, y_test))
print("随机森林回归的R_squared值为：", r2_score(y_test, rfr_y_predict))
print("随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                          ss_y.inverse_transform(rfr_y_predict)))
print("随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                             ss_y.inverse_transform(rfr_y_predict)))

# 极端随机森林回归模型评估
print("极端随机森林回归的默认评估值为：", etr.score(x_test, y_test))
print("极端随机森林回归的R_squared值为：", r2_score(y_test, gbr_y_predict))
print("极端随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                            ss_y.inverse_transform(gbr_y_predict)))
print("极端随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(gbr_y_predict)))

# 梯度提升回归模型评估
print("梯度提升回归回归的默认评估值为：", gbr.score(x_test, y_test))
print("梯度提升回归回归的R_squared值为：", r2_score(y_test, etr_y_predict))
print("梯度提升回归回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                            ss_y.inverse_transform(etr_y_predict)))
print("梯度提升回归回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(etr_y_predict)))

'''
随机森林回归的默认评估值为： 0.91718814102
随机森林回归的R_squared值为： 0.91718814102
随机森林回归的均方误差为: 0.237025968679
随机森林回归的平均绝对误差为: 0.287554716981
极端随机森林回归的默认评估值为： 0.847843008968
极端随机森林回归的R_squared值为： 0.91718814102
极端随机森林回归的均方误差为: 0.237025968679
极端随机森林回归的平均绝对误差为: 0.287554716981
梯度提升回归回归的默认评估值为： 0.671650998451
梯度提升回归回归的R_squared值为： 0.91718814102
梯度提升回归回归的均方误差为: 0.237025968679
梯度提升回归回归的平均绝对误差为: 0.287554716981
'''

