import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


oil = pd.read_csv(r'data/BrentOilPrices.csv')
case = pd.read_excel(r'data/31汽车上市公司诉讼仲裁数据(200308-202303).xlsx')
#print(oil.head())
#print(oil.tail())
#数据处理 -- 时间数据格式 -- year

date = oil['Date'].astype(str).values.tolist()

for i in range(0,len(date)):
    date[i]=date[i][-2:]
oil['Year'] = date
#print(pd.value_counts(oil['Year']))
#print(oil.head())

date=date.clear()
date = case['公告日期'].astype(str).values.tolist()
for i in range(0,len(date)):
    date[i]=date[i].strip()[:5]
case['Year'] = date
#print(case.head())
#计算平均油价
oilprice = oil.groupby(['Year'],as_index=False).agg({"Price":np.mean})
oilprice_year = []
for year,price in oilprice[['Year','Price']].values:
    if '22' >= year >= '10':
        oilprice_year.append(price)

print("oil")
print(oilprice_year)
print(len(oilprice_year))
#平均案件数

case = pd.read_excel(r'data/31汽车上市公司诉讼仲裁数据(200308-202303).xlsx',header=0)
## 同样的时间格式处理
date=[]
date = case['公告日期'].astype(str).values.tolist()
for i in range(0,len(date)):
    date[i]=date[i].strip()[:4]
case['Year'] = date

#平均案件数
cases= case.groupby(['Year'],as_index=False)['序号'].count()


case_year =[]
for year,cnt in cases[['Year','序号']].values:
    if '2010'<=year<='2022':
        case_year.append(cnt)

print("cases:")
print(case_year)
print(len(case_year))
#平均车价
car = pd.read_csv(r'data/Electric cars.csv',nrows=13)
carprice_year = car['Average price of new car'].values.tolist()
print("car:")
print(carprice_year)
print(len(carprice_year))

#充电桩数量

charge_df = pd.read_excel(r'data/2016-2022公共充电桩（直流、交流、交直流一体）数量.xlsx',skiprows=1,nrows=8)
charge = charge_df['公共类充电桩数量（万）'].values.tolist()
charge = [0,2,4,6,8,10] + charge
print("charge:")
print(charge)
print(len(charge))

#专利数量
pattern_data = pd.read_excel(r'data/pattern/21汽车上市公司专利数量-发明专利（年，1999-2022，68家）.xls',skiprows=6,header=None)
pattern = pattern_data.iloc[:,1:].mean(axis=1,skipna=True)[12::-1].values.tolist()
print("pattern:")
print(pattern)
print(len(pattern))

#销量
new_energy = pd.read_excel(r'2010_/2010-2022新能源汽车产量、销量.xlsx',nrows=14)
productio_year = new_energy['销量'].values.tolist()
print("sale:")
print(productio_year)
print(len(productio_year))

data = pd.DataFrame()
data['EVSale'] = productio_year
data['Case']=case_year
data['OilPrice'] = oilprice_year
data ['EVPrice'] = carprice_year
data['Charge'] = charge
data['Pattern'] = pattern


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
print(scaled_df)
#select
X = scaled_df.iloc[:,1:].values
y = scaled_df.iloc[:,0].values
print(y)


from sklearn.linear_model import LinearRegression,Ridge,Lasso
#Linear
Linearmodel = LinearRegression()
Linearmodel.fit(X,y)
y_pred_linear = Linearmodel.predict(X)
#rudge
ridge = Ridge(alpha=0.1)
ridge.fit(X,y)
y_pred_ridge = ridge.predict(X)

#Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X,y)
y_pred_lasso = lasso_model.predict(X)

from sklearn.metrics import mean_squared_error,r2_score


#jieguo
Lin_coef = Linearmodel.coef_
print(Lin_coef)
Gri_coef = ridge.coef_
print(Gri_coef)
Lossa_coef = lasso_model.coef_
print(Lossa_coef)

"""
import matplotlib.pyplot as plt
# 可视化
plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y_pred_linear, color='blue', linestyle='-', marker='o', label='Linear Regression')
plt.plot(range(len(y)), y_pred_ridge, color='orange', linestyle='--', marker='x', label='Ridge Regression')
plt.plot(range(len(y)), y_pred_lasso, color='green', linestyle='-.', marker='s', label='Lasso Regression')
plt.scatter(range(len(y)), y, color='red', label='Original Data',s=50)
plt.xlabel('Sample Index')
plt.ylabel('Standardized Values')
plt.title('Comparison of Regression Models')
plt.legend()
plt.savefig('regression.png')
plt.show()
"""