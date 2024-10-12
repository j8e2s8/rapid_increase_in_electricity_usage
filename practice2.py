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
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project2/data_week2.csv

# 컬럼명 바꾸기
df.columns  # ['num', 'date_time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)','강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유']
df = df.rename(columns = {'num' : '건물번호', 'date_time' : '날짜' , '전력사용량(kWh)' : '전력사용량' , '기온(°C)':'기온', '풍속(m/s)' :'풍속'  , '습도(%)':'습도' , '강수량(mm)':'강수량', '일조(hr)' : '일조'  })
df.head()


# 데이터 자료형 바꾸기
#df['건물번호'] = df['건물번호'].astype('object')
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
df.drop('시', axis=1, inplace=True)  # '시' 칼럼 제거

# 평일/주말 구분
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말: 1, 평일: 0

# 평일(0값)인 행 확인
weekday_data = df[df['주말'] == 0]
weekday_data

# 주말(1값)인 행 확인
weekend_data = df[df['주말'] == 1]
weekend_data


# 3개 행씩 묶어 평균 계산
df_avg = df.groupby(df.index // 3).mean(numeric_only=True)
df_avg['날짜'] = df.groupby(df.index // 3)['날짜'].first().values

# 군집화
buildings = df_avg['건물번호'].unique()
series_data = [df_avg[df_avg['건물번호'] == bld]['전력사용량'].values for bld in buildings]

scaler = TimeSeriesScalerMeanVariance()
series_data_scaled = scaler.fit_transform(series_data)

n_clusters = 5  # 원하는 군집 수 지정
dtw_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
labels = dtw_kmeans.fit_predict(series_data_scaled)

result_df = pd.DataFrame({'건물번호': buildings, '군집': labels})
df_avg = df_avg.merge(result_df[['건물번호', '군집']], on='건물번호', how='left')

# 군집화 결과 시각화
for cluster_num in range(n_clusters):
    plt.figure(figsize=(10, 6))
    plt.title(f'Cluster {cluster_num} 전력사용량 패턴')
    for i, building in enumerate(buildings):
        if labels[i] == cluster_num:
            plt.plot(series_data[i], label=f'건물 {building}')
    plt.xlabel('시간')
    plt.ylabel('전력사용량')
    plt.legend(loc='upper right')
    plt.show()


# 군집 번호에 따른 데이터 반환 함수
def cluster(cluster_label):
    # 지정된 군집 번호로 필터링
    df_cluster = df_avg.query(f'군집 == {cluster_label}')

    # 이전 4개의 동일 요일, 시간대 전력 사용량의 중앙값을 구해 새로운 열에 추가
    median_series = (
        df_cluster.groupby(['건물번호','요일', '시간_sin'])['전력사용량'] # 건물번호 추가?
        .apply(lambda x: x.shift().rolling(window=4, min_periods=1).median())
    )

    median_df = pd.DataFrame({'index' : median_series.index.get_level_values(3),
                               '전력중앙값' : median_series.values})

    df_cluster = pd.merge(df_cluster, median_df, how='left', left_index=True, right_on='index')
    df_cluster.set_index('index', inplace=True)

    # 변화율 계산하여 새로운 칼럼 추가
    df_cluster['변화율'] = ((df_cluster['전력사용량'] - df_cluster['전력중앙값']) / df_cluster['전력중앙값']) * 100

    df_cluster = df_cluster.dropna()

    # 급증 기준: 동일 요일 동 시간대 4개의 중앙값 대비 증가율 30% 초과
    df_cluster['급증'] = df_cluster['변화율'] > 30

    # 전역 변수로 할당 (원하는 경우)
    globals()[f'df_{cluster_label}'] = df_cluster

# 빌딩 1부터 60까지의 급증 갯수 계산
surge = []
for i in range(1, 61):
    building(i)
    df_n = globals()[f'df_{i}']
    surge_count = df_n['급증'].sum()
    surge.append({'건물번호': i, '급증갯수': surge_count})

# 결과를 데이터프레임으로 변환
surge_df = pd.DataFrame(surge)
surge_df.sort_values(by='급증갯수',ascending=False)

# building(4)
# df_4['급증'].sum()
# df_4['변화율'].describe()



# 냉방, 태양광 여부 별로 급증 개수 파악
result = [f'df_{i+1}' for i in range(60)]
df_list = [globals()[name] for name in result]
total = pd.concat(df_list, ignore_index=True)

group_df = total.groupby(['비전기냉방설비운영','태양광보유']).agg(급증_개수 = ('급증', 'sum'), 데이터수 = ('급증','count'))
group_df['비율'] = group_df['급증_개수']/group_df['데이터수']
group_df





# 건물별 전력사용량 데이터 생성
buildings = df_avg['건물번호'].unique()
series_data = [df_avg[df_avg['건물번호'] == bld]['전력사용량'].values for bld in buildings]

# 시계열 데이터 전처리 (표준화)
scaler = TimeSeriesScalerMeanVariance()
series_data_scaled = scaler.fit_transform(series_data)

# DTW K-평균 군집화
n_clusters = 5  # 원하는 군집 수 지정
dtw_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
labels = dtw_kmeans.fit_predict(series_data_scaled)

# 결과를 데이터프레임으로 정리
result_df = pd.DataFrame({'건물번호': buildings, '군집': labels})
result_df['군집'].unique()








# 전력사용량 예측을 위한 입력과 출력 준비
# LSTM을 위한 시퀀스 데이터 생성
sequence_length = 2  # 시퀀스 길이 설정 (예: 2시간)
X, y = [], []

for i in range(len(df) - sequence_length):
    X.append(df.iloc[i:i + sequence_length, 1:].values)  # 전력 사용량 제외한 모든 특성
    y.append(df.iloc[i + sequence_length, 0])  # 다음 시점의 전력 사용량

X = np.array(X)
y = np.array(y)

# 데이터 정규화
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 훈련과 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, X.shape[2])))  # 시퀀스 길이와 피쳐 수
model.add(Dropout(0.2))  # 드롭아웃 레이어 추가
model.add(Dense(1))  # 출력층 (회귀 문제이므로 활성화 함수 없음)

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model.fit(X_train, y_train, epochs=20, batch_size=1)

# 모델 평가
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')

# 예측
predictions = model.predict(X_test)
print(predictions)













# 건물_4에 대한 급증 구간 그래프
highlight = df_4['전력사용량'] > 1500  # 표시하고 싶은 필터링 구간

plt.figure(figsize=(12, 6))
plt.plot(df_4['날짜'], df_4['전력사용량'], label='전력 사용량', color='blue')
plt.fill_between(df_4['날짜'], df_4['전력사용량'], where=highlight, color='red', alpha=0.5, label='급증 구간')
plt.title('전력 사용량 시계열 그래프')
plt.xlabel('날짜')
plt.ylabel('전력 사용량')
plt.legend()
plt.show()






# pip install tensorflow imbalanced-learn
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split


X = df[['전력사용량', '기온', '풍속', '습도', '강수량', '일조', '비전기냉방설비운영', '태양광보유']].values
y = df['급증'].values

# 데이터 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE를 사용하여 클래스 불균형 해결
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# LSTM 입력 형식으로 변환 (3D 텐서)
X_reshaped = X_resampled.reshape((X_resampled.shape[0], 1, X_resampled.shape[1]))

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_resampled, test_size=0.2, random_state=42)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

# 예측
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # 0.5 이상이면 1로 변환

# 예측 결과 출력
print(predictions)