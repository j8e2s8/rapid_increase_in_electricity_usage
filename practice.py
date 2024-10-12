import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc 
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
import lightgbm as lgb


# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project2/data_week2.csv

# 컬럼명 바꾸기
df.columns  # ['num', 'date_time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)','강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유']
df = df.rename(columns = {'num' : '건물번호', 'date_time' : '날짜' , '전력사용량(kWh)' : '전력사용량' , '기온(°C)':'기온', '풍속(m/s)' :'풍속'  , '습도(%)':'습도' , '강수량(mm)':'강수량', '일조(hr)' : '일조'  })
df.head()





def kde(df, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.kdeplot(data=df, x=col, fill=True , palette=palette, alpha=alpha)
		plt.title(f'{col}의 확률밀도', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


df.columns
df.head()

df.info()


df['건물번호'] = df['건물번호'].astype('object')
df['날짜'] = pd.to_datetime(df['날짜'])
df['비전기냉방설비운영'] = df['비전기냉방설비운영'].astype('boolean')
df['태양광보유'] = df['태양광보유'].astype('boolean')

# 날짜 인코딩
df['년'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek  # 0: 월요일, 6: 일요일
df['시'] = df['날짜'].dt.hour
df['시간_sin'] = np.sin(2 * np.pi * df['시'] / 24)
df['시간_cos'] = np.cos(2 * np.pi * df['시'] / 24)
# df.drop('시', axis=1, inplace=True)  # '시' 칼럼 제거



# 평일/주말 구분
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말: 1, 평일: 0


kde(df)

df.describe()
df[['비전기냉방설비운영','태양광보유']].unique()

df.iloc[122362:122373,:]

df.info()

df[(df['월']==8) & (df['일']==24) & (df['시']==3)&(df['건물번호']==1)]
df[(df['월']==8) & (df['일']==17) & (df['시']==3)&(df['건물번호']==1)]
df[(df['월']==8) & (df['일']==10) & (df['시']==3)&(df['건물번호']==1)]
df[(df['월']==8) & (df['일']==3) & (df['시']==3)&(df['건물번호']==1)]
df[(df['월']==7) & (df['일']==27) & (df['시']==3)&(df['건물번호']==1)]

df[(df['월']==8) & (df['일']==23) & (df['시']==3)&(df['건물번호']==1)]
df[(df['월']==8) & (df['일']==22) & (df['시']==3)&(df['건물번호']==1)]


df[((df['월']==7) & (df['일']==8)& ((df['시']==6)|(df['시']==7)|(df['시']==8))) & (df['건물번호']==26)]
df[((df['월']==7) & (df['일']==15)& ((df['시']==6)|(df['시']==7)|(df['시']==8))) & (df['건물번호']==26)]
df[((df['월']==7) & (df['일']==22)& ((df['시']==6)|(df['시']==7)|(df['시']==8))) & (df['건물번호']==26)]
df[((df['월']==7) & (df['일']==29)& ((df['시']==6)|(df['시']==7)|(df['시']==8))) & (df['건물번호']==26)]
df[((df['월']==8) & (df['일']==5)& ((df['시']==6)|(df['시']==7)|(df['시']==8))) & (df['건물번호']==26)]




np.mean([2227.9536, 2393.4528, 2810.8944])
np.mean([2298.3912, 2480.9976, 2978.2728])
np.mean([2368.8288, 2568.5424, 3145.6512])
np.mean([2439.2664, 2656.0872, 3313.0296])
np.mean([2509.704, 2743.632, 3480.408])


for i in df.columns:
	print(f'{i}컬럼의 unique 개수 :',len(df[i].unique()))
      

cols = ['건물번호','비전기냉방설비운영','태양광보유']
for i in cols:
	print(f'{i}컬럼의 unique :', df[i].unique())
df['']


# 수정하기
df= dff
y='기온'

def timeline(df, y, x_time_n=None, x_time_s = 'MS' ,palette='dark'):
	col = df.select_dtypes(include=['boolean','object']).columns
	w_n = sum([len(df[i].unique()) for i in col])
	n = int(np.ceil(w_n/4))
	u = []
	for i in col:
		for v in df[i].unique():
			u.append([i, v])
	plt.clf()
	plt.figure(figsize=(6*4, 5*n))
	for index, col_u in enumerate(u, 1): 
		# plt.figure(figsize=(6*4, 5*n))
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		df2 = df[df[col_u[0]] == col_u[1]]
		sns.lineplot(data=df2, x='날짜', y=y, palette=palette)
		if x_time_n is None:
			plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=x_time_n))  # 눈금 개수 조정
		else:
			monthly_ticks = pd.date_range(start=df['날짜'].min(), end=df['날짜'].max(), freq=x_time_s)
			plt.xticks(monthly_ticks, monthly_ticks.strftime('%Y-%m-%d'), rotation=45)
		plt.title(f'{col_u[0]}의 {col_u[1]}범주에 대한 확률밀도', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



timeline(df[(df['월']==8) & ((df['일']>=3)&(df['일']<=9)) & (df['건물번호']==26)],'전력사용량', x_time_s='D')


plt.figure(figsize=(10,7))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df2 = df[(df['월']==8) & ((df['일']>=3)&(df['일']<=9)) & (df['건물번호']==26)]
sns.lineplot(data=df2, x='날짜', y='전력사용량')
sns.scatterplot(data=df2, x='날짜', y=df2['전력사용량'],color='red')
monthly_ticks = pd.date_range(start=df2['날짜'].min(), end=df2['날짜'].max(), freq='D')
# 날짜와 요일을 함께 표시
tick_labels = [f"{date.strftime('%Y-%m-%d')}\n{date.strftime('%a')}" for date in monthly_ticks]
plt.xticks(monthly_ticks, tick_labels)



plt.figure(figsize=(10,7))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df2 = df[(df['월']==8) & ((df['일']>=18)&(df['일']<=24)) & (df['건물번호']==26)]
sns.lineplot(data=df2, x='날짜', y='전력사용량')
sns.scatterplot(data=df2, x='날짜', y=df2['전력사용량'],color='red')
monthly_ticks = pd.date_range(start=df2['날짜'].min(), end=df2['날짜'].max(), freq='D')
# 날짜와 요일을 함께 표시
tick_labels = [f"{date.strftime('%Y-%m-%d')}\n{date.strftime('%a')}" for date in monthly_ticks]
plt.xticks(monthly_ticks, tick_labels)

highlight = df2['날짜'] == '2020-08-24 10:00:00'
plt.figure(figsize=(10,7))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df3 = df[(df['월']==8) & ((df['일']==3)|(df['일']==10)|(df['일']==17)) & (df['건물번호']==26)]
sns.lineplot(data=df2, x='날짜', y='전력사용량')
sns.scatterplot(data=df2, x='날짜', y=df2['전력사용량'],color='red')
plt.fill_between(df2['날짜'], df2['전력사용량'], where=highlight, color='red', alpha=0.5, label='급증 구간')
monthly_ticks = pd.date_range(start=df2['날짜'].min(), end=df2['날짜'].max(), freq='D')
# 날짜와 요일을 함께 표시
tick_labels = [f"{date.strftime('%Y-%m-%d')}\n{date.strftime('%a')}" for date in monthly_ticks]
plt.xticks(monthly_ticks, tick_labels)
plt.ylim([2000,6000])


df3 = df[(((df['월']==7) & (df['일']==27)) | ((df['월']==8) & ((df['일']==3)|(df['일']==10)|(df['일']==17)))) & (df['건물번호']==26)]
df4 = df[((df['월']==8) & (df['일']==24)) & (df['건물번호']==26)]
df4[['날짜','전력사용량']].sort_values('전력사용량')






highlight = df3['날짜'] == '2020-08-05 9:00:00'
plt.figure(figsize=(10,7))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df3 = df[(df['월']==8) & ((df['일']>=3)&(df['일']<=9)) & (df['건물번호']==26)]
sns.lineplot(data=df3, x='날짜', y='전력사용량', color='black')
sns.scatterplot(data=df3, x='날짜', y=df3['전력사용량'],color='red', s=30)
sns.scatterplot(x=[pd.to_datetime('2020-08-05 09:00:00')], y=[4514.94], color='#FF00FF', s=70)
sns.scatterplot(x=[pd.to_datetime('2020-08-05 09:00:00')], y=[4106.7], color='green', s=70)
plt.fill_between(df3['날짜'], df3['전력사용량'], where=highlight, color='#40E0D0', label='급증 구간')
monthly_ticks = pd.date_range(start=df3['날짜'].min(), end=df3['날짜'].max(), freq='D')
tick_labels = [f"{date.strftime('%Y-%m-%d')}\n{date.strftime('%a')}" for date in monthly_ticks]
plt.xticks(monthly_ticks, tick_labels)
plt.ylim([2000,6000])

df[(df['월']==8) & (df['일']==5) & (df['시'] ==9) & (df['건물번호']==26)]



df[(df['월']==7) & (df['일']==29) & (df['시'] ==9) & (df['건물번호']==26)]
df[(df['월']==7) & (df['일']==22) & (df['시'] ==9) & (df['건물번호']==26)]
df[(df['월']==7) & (df['일']==15) & (df['시'] ==9) & (df['건물번호']==26)]
df[(df['월']==7) & (df['일']==8) & (df['시'] ==9) & (df['건물번호']==26)]

df[(df['월']==8) & (df['일']==4) & (df['시'] ==9) & (df['건물번호']==26)]
df[(df['월']==8) & (df['일']==3) & (df['시'] ==9) & (df['건물번호']==26)]

df[(df['월']==8) & (df['일']==5) & (df['시'] ==8) & (df['건물번호']==26)]
df[(df['월']==8) & (df['일']==5) & (df['시'] ==7) & (df['건물번호']==26)]






timeline(df, '전력사용량')
timeline(df, '기온')
timeline(df, '풍속')
timeline(df, '습도')
timeline(df, '강수량')
timeline(df, '일조')

df.columns
dff=df[(df['월']==8) & ((df['일']==23) | (df['일']==24))]
timeline(dff, '기온')
timeline(df[(df['월']==8) & ((df['일']==23) | (df['일']==24))], '풍속')




timeline(df[(df['월']==8) & ((df['일']==23) | (df['일']==24))], '기온', x_time_n = 5)



k= df[(df['월']==8) & (df['일']==23) | (df['일']==24)]




df[(df['월']==8) & (df['일']==23) | (df['일']==24)]
df.columns
df[df['월']==6]['일'].unique()

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# plt.subplot(n, 4, index)
# dfn = df[df[col_u[0]] == col_u[1]]
sns.lineplot(data=df, x='날짜', y='기온')
monthly_ticks = pd.date_range(start=df['날짜'].min(), end=df['날짜'].max(), freq='MS')
plt.xticks(monthly_ticks, monthly_ticks.strftime('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 눈금 개수 조정
plt.title('기온의 대한 확률밀도', fontsize=20)


import seaborn as sns
df2day = df[(df['월']==8) & ((df['일']>=18))]
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# plt.subplot(n, 4, index)
# dfn = df[df[col_u[0]] == col_u[1]]
sns.lineplot(data=df2day, x='날짜', y='강수량')
monthly_ticks = pd.date_range(start=df2day['날짜'].min(), end=df2day['날짜'].max(), freq='D')
plt.xticks(monthly_ticks, monthly_ticks.strftime('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 눈금 개수 조정
plt.title('풍속의 대한 확률밀도', fontsize=20)





# 월별 평균 계산
df2=df.copy()
df2.set_index('날짜', inplace=True)

monthly_mean = df2.resample('M').mean()


# 주기적인 패턴 시각화
plt.figure(figsize=(12, 6))
plt.plot(monthly_mean.index, monthly_mean, label='Monthly Average')
plt.title('Monthly Average Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.legend()
plt.grid()
plt.show()





# STL 분해 : 인덱스를 날짜 컬럼으로 해야 함
from statsmodels.tsa.seasonal import STL
df2.index = pd.to_datetime(df2.index)
stl = STL(df2['전력사용량'], seasonal=24)  # seasonal 주기를 설정
result = stl.fit()

# 결과 시각화
fig = result.plot()
plt.show()









df['z_scores'] = (df['전력사용량(kWh)'] - df['전력사용량(kWh)'].mean()) / np.std(df['전력사용량(kWh)'])
kde(df)


group1 = df[(df['비전기냉방설비운영']==1) & (df['태양광보유']==1)]
group2 = df[(df['비전기냉방설비운영']==0) & (df['태양광보유']==1)]
group3 = df[(df['비전기냉방설비운영']==1) & (df['태양광보유']==0)]
group4 = df[(df['비전기냉방설비운영']==0) & (df['태양광보유']==0)]

 
kde(group1)
kde(group2)
kde(group3)
kde(group4)


for i in df.columns:
      df[df.columns == i]



# [boxcox] - X4 , X13, X18 , X19 , X20 제거한 df2의 box-cox
box_add_df = df.copy()
boxcox_vars = []
df

for column in df.columns[:-1]:
    transformed_variable, lambda_value = stats.boxcox(df[column] + 1)
    box_add_df[f'boxcox_{column}'] = transformed_variable
    boxcox_vars.append((column, lambda_value))

for original, lambda_val in boxcox_vars:
    print(f"{original}의 최적의 Box-Cox 변환 λ 값: {lambda_val}")

box_add_df.columns
np.where(box_add_df.columns == 'Y')
box_df = box_add_df.iloc[:,15:]

# box_add_df 데이터셋 : 기존에 있는 변수 + boxcox 변환한 변수 (boxcox_어쩌구) 포함한 데이터셋
# box_df 데이터셋 : boxcox 변환한 변수 (boxcox_어쩌구)만 포함한 데이터셋


df[(df['월'] == 8) & (df['일'] == 24) & ((df['건물번호'] ==1) | (df['건물번호'] ==2))]


df.groupby('날짜').agg(
    기온_std=('기온', 'std'),
    풍속_std=('풍속', 'std'),
    습도_std=('습도', 'std'),
    강수량_std=('강수량', 'std')
)



# --------------------------

## 건물1
df_1=df.query('건물번호 == 1')
df_1.head()

# 이전 4개의 동일 요일, 시간대 전력 사용량의 중앙값을 구해 새로운 열에 추가
# 그룹별 중앙값 계산
median_series = (
    df_1.groupby(['요일', '시간_sin'])['전력사용량']
    .apply(lambda x: x.shift().rolling(window=4, min_periods=1).median())
)

median_series.index.get_level_values(2)
median_series.values

median_df = pd.DataFrame({'index' : median_series.index.get_level_values(2),
							'value' : median_series.values})
median_df.isna().sum()

df_2 = pd.merge(df_1, median_df, how='left', left_index=True, right_on='index')
df_2.set_index('index', inplace=True)




df_avg = df.groupby(df.index // 3).mean(numeric_only=True)
df_avg['날짜'] = df.groupby(df.index // 3)['날짜'].first().values




def building(n):
    # 지정된 건물 번호로 필터링
    df_n = df_avg.query(f'건물번호 == {n}')

    # 이전 4개의 동일 요일, 시간대 전력 사용량의 중앙값을 구해 새로운 열에 추가
    median_series = (
        df_n.groupby(['요일', '시간_sin'])['전력사용량']
        .apply(lambda x: x.shift().rolling(window=4, min_periods=1).median())
    )

    median_df = pd.DataFrame({'index' : median_series.index.get_level_values(2),
                        '전력중앙값' : median_series.values})

    df_n = pd.merge(df_n, median_df, how='left', left_index=True, right_on='index')
    df_n.set_index('index', inplace=True)


    # 변화율 계산하여 새로운 칼럼 추가
    df_n['변화율'] = ((df_n['전력사용량'] - df_n['전력중앙값']) / df_n['전력중앙값']) * 100

    # 급증 기준: 동일 요일 동 시간대 4개의 중앙값 대비 증가율 30% 초과
    df_n['급증'] = df_n['변화율'] > 30

    # 전역 변수로 할당
    globals()[f'df_{n}'] = df_n
    globals()[f'train_df_{n}'] = train_df_n
    globals()[f'test_df_{n}'] = test_df_n



for i in range(60):
	building(i+1)

result = [f'df_{i+1}' for i in range(60)]
df_list = [globals()[name] for name in result]
pd.concat(df_list, ignore_index=True)


df_4

kde(df_1)
kde(df_2)



box_add_df = df_4.copy()
boxcox_vars = []



df_num = df_11  # 얘만 바꾸기

transformed_variable, lambda_value = stats.boxcox(df_num['변화율'] + 1)
df_num[f'boxcox_전력사용량'] = transformed_variable
print(f"전력사용량의 최적의 Box-Cox 변환 λ 값: {lambda_val}")










df_4
# 급증하는 구간 정의 (예: 전력 사용량이 1500 이상인 경우)
highlight = df_4['전력사용량'] > 1500  # 표시하고 싶은 필터링 구간

plt.figure(figsize=(12, 6))
plt.plot(df_4['날짜'], df_4['전력사용량'], label='전력 사용량', color='blue')
plt.fill_between(df_4['날짜'], df_4['전력사용량'], where=highlight, color='red', alpha=0.5, label='급증 구간')
plt.title('전력 사용량 시계열 그래프')
plt.xlabel('날짜')
plt.ylabel('전력 사용량')
plt.legend()
plt.show()
