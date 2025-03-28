# 시각화와 회귀 예측 개념 정리를 위한 실험 파일일

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats


# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:\Users\USER\Documents\D\LS 빅데이터 스쿨 3기\rapid_increase_in_electricity_usage/data_week2.csv

# 컬럼명 바꾸기
df.columns  # ['num', 'date_time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)','강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유']
df = df.rename(columns = {'num' : '건물번호', 'date_time' : '날짜' , '전력사용량(kWh)' : '전력사용량' , '기온(°C)':'기온', '풍속(m/s)' :'풍속'  , '습도(%)':'습도' , '강수량(mm)':'강수량', '일조(hr)' : '일조'  })
df.head()


df.info()
num_df = df.select_dtypes('float').iloc[:,:-3]


# 피어슨 상관관계 
pearson_m = num_df.corr()
spear_m = num_df.corr('spearman')

# 히트맵 시각화
plt.figure(figsize=(10,8))
sns.heatmap(pearson_m , annot=True, fmt=".2f" , cmap="coolwarm", square=True, cbar_kws={'shrink': .8}, annot_kws={'size':15})
					# annot=True : 각 셀에 값을 표시해줌.  # fmt=".2f" : 숫자를 소수점 2자리까지 표시   # cmap : 팔레트 지
					# square=True : 각 셀의 비율을 정사각형으로 함  # cbar_kws : 색상바 옵션 {'shrink': .8} 이면 색상바를 80%로 줄
					# annot_kws : 셀에 표시될 텍스트 크기 조정 옵션. {'size': 8} 이면 8 크기로 설정
plt.title('pearson correlation marix', fronsize=14)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결
plt.xticks(rotation=45, ha='right', fontsize=10)  # ha='right' : 오른쪽 정
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

#쌍플롯
sns.pairplot(num_df)
plt.show()

sns.pairplot(df)
plt.show()

hue_df = pd.concat([num_df, df['건물번호']], axis=1)

sns.pairplot(num_df, kind='reg')  # 산점도에 회귀선 추가
plt.show()

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결
sns.pairplot(hue_df, hue='건물번호')  # 산점도에 회귀선 추가
plt.show()

sns.pairplot(df, kind='reg', hue='건물번호')  # 산점도에 회귀선 추가
plt.show()

sns.pairplot(num_df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})  # 회귀선 색상 'green'
plt.show()

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결 
sns.pairplot(hue_df, hue='건물번호', kind='reg', plot_kws={'line_kws': {'color': 'red'}})  # 회귀선 색상 'red'
plt.show()

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결 
sns.pairplot(num_df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})  # 회귀선 색상 'red'
plt.show()

df2 = pd.DataFrame({'성별':[0,0,0,0,0,1], '키':[170,165,191,174,180,163]})
try_corr = df2.corr()


# 네트워크 플롯
import networkx as nx
import matplotlib.pyplot as plt

# 빈 그래프 생성
G = nx.Graph()

# 노드 추가
G.add_node(1)
G.add_node(2)
G.add_node(3)

# 엣지(간선) 추가
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(1, 3)

# 그래프 그리기
nx.draw(G, with_labels=True, node_size=700, node_color='lightblue', font_size=15)

# 그래프 출력
plt.show()



# 
import matplotlib.font_manager as fm
pearson_m = num_df.corr()

# 폰트 경로를 올바르게 설정 (맑은 고딕 폰트 경로)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 경우
prop = fm.FontProperties(fname=font_path)

# matplotlib의 기본 폰트 설정
plt.rcParams['font.family'] = prop.get_name()  # 한글 폰트 설정
G = nx.Graph()

# 상관 계수가 0.2 이상인 것들만 엣지로 추가
threshold = 0.2
for i in range(len(pearson_m.columns)):
    for j in range(i):
        if abs(pearson_m.iloc[i, j]) > threshold:  # 상관 계수 임계값 적용
            G.add_edge(pearson_m.columns[i], pearson_m.columns[j], weight=pearson_m.iloc[i, j])

# 네트워크 그래프 그리기
pos = nx.spring_layout(G)  # 레이아웃 설정
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12)
#pos = nx.circular_layout(G)
#nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# 그래프 출력
plt.title("Correlation Network")
plt.show()





# test set 만들기기
try_df = df[['습도','기온']]
X_train, X_test, y_train, y_test = train_test_split(try_df['기온'], try_df['습도'], test_size=0.25, random_state=20250313)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

train = pd.DataFrame({'기온':X_train, '습도':y_train})
test = pd.DataFrame({'기온':X_test, '습도':y_test})


# 평균으로 y값 예측
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결 
sns.scatterplot(data=train, y='습도', x='기온', alpha=0.2, color='blue')
sns.scatterplot(data=test, y='습도', x='기온', alpha=0.2 , color='orange')
plt.axhline(y= np.mean(train['습도']), color='green')
plt.axhline(y= np.mean(test['습도']), color='red')
plt.text(22, 81, 'train_y_bar (y_pred) = 80.13', ha='center', va='bottom', size=13, color='green')
plt.text(22, 73, 'test_y_bar (test R^2에 사용될 평균값) = 80.28', ha='center', va='bottom', size=13, color='red')
plt.show()


mean_pred_r2 = r2_score(y_test, np.repeat(np.mean(train['습도']),30600))
# -9.46732987630039e-05

sst = np.sum((y_test - np.mean(y_test)) ** 2)

# SSR (Regression Sum of Squares)
ssr = np.sum((np.repeat(np.mean(train['습도']),30600) - np.mean(y_test)) ** 2)

# SSE (Error Sum of Squares)
sse = np.sum((y_test - np.repeat(np.mean(train['습도']),30600)) ** 2)

ssr/sst


model = LinearRegression()
model.fit(np.array(train['기온']).reshape(-1,1), train['습도'])  # train x, y로 모델을 적합시키기. 즉, 데이터에 알맞은 직선의 기울기, 절편 값 구하기.(=식 구하기)				
test_y_pred = model.predict(np.array(test['기온']).reshape(-1,1))  # train x로 예측 y값 구하기.   # predict는 β0_hat + β1_hat * x1 으로 대응되는 y값을 구한 것임. 

# 회귀로 y값 예측
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 해결 
sns.scatterplot(data=train, y='습도', x='기온', alpha=0.2, color='blue')
sns.scatterplot(data=test, y='습도', x='기온', alpha=0.2 , color='orange')
plt.plot(test['기온'],test_y_pred, color='green', label='train으로 학습한 회귀직선 (test_y_pred)')
plt.axhline(y= np.mean(test['습도']), color='red')
# plt.text(22, 81, 'train_y_bar (y_pred) = 80.13', ha='center', va='bottom', size=13, color='green')
plt.text(22, 73, 'test_y_bar (test R^2에 사용될 평균값) = 80.28', ha='center', va='bottom', size=13, color='red')
plt.legend()
plt.show()

reg_r2 = r2_score(y_test, test_y_pred)
# 0.25546048434064283

sst = np.sum((y_test - np.mean(y_test)) ** 2)

# SSR (Regression Sum of Squares)
ssr = np.sum((test_y_pred - np.mean(y_test)) ** 2)

# SSE (Error Sum of Squares)
sse = np.sum((y_test - test_y_pred) ** 2)

ssr/sst
1-(sse/sst)
adjusted_r2 = 1 - (1 - reg_r2) * (30600 - 1) / (30600 - 1 - 1)

