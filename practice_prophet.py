from prophet import Prophet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from catboost import CatBoostClassifier


# 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project2/data_week2.csv

df = df.rename(columns={
    'num': '건물번호', 'date_time': '날짜', '전력사용량(kWh)': '전력사용량',
    '기온(°C)': '기온', '풍속(m/s)': '풍속', '습도(%)': '습도', '강수량(mm)': '강수량', '일조(hr)': '일조'
})

# 데이터 자료형 변환
df['날짜'] = pd.to_datetime(df['날짜'])
df['비전기냉방설비운영'] = df['비전기냉방설비운영'].astype('boolean')
df['태양광보유'] = df['태양광보유'].astype('boolean')

# 날짜 인코딩
#df['년'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek  # 0: 월요일, 6: 일요일
df['시'] = df['날짜'].dt.hour
#df['시간_sin'] = np.sin(2 * np.pi * df['시'] / 24)
#df['시간_cos'] = np.cos(2 * np.pi * df['시'] / 24)

df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말: 1, 평일: 0

# 파생변수 추가
df['기온_24시간평균'] = df['기온'].rolling(window=24).mean()  # 최근 24시간 평균
# df['일조_24시간평균'] = df['일조'].rolling(window=24).mean()  # 최근 24시간 평균

# PolynomialFeatures 추가
poly_features = df[['기온', '풍속', '습도']]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_transformed = poly.fit_transform(poly_features)
poly_columns = poly.get_feature_names_out(['기온', '풍속', '습도'])
poly_df = pd.DataFrame(poly_transformed, columns=poly_columns)
poly_df.drop(['기온', '풍속', '습도'], axis=1, inplace=True)
df = pd.concat([df, poly_df], axis=1)

# 건물 번호에 따른 데이터 반환 함수
def building(n):
    # 지정된 건물 번호로 필터링
    df_n = df.query(f'건물번호 == {n}')

    # 이전 4개의 동일 요일, 시간대 전력 사용량의 중앙값을 구해 새로운 열에 추가
    median_series = (
        df_n.groupby(['요일', '시'])['전력사용량']
        .apply(lambda x: x.shift().rolling(window=4, min_periods=1).median())
    )

    median_df = pd.DataFrame({'index' : median_series.index.get_level_values(2),
                        '전력중앙값' : median_series.values})

    df_n = pd.merge(df_n, median_df, how='left', left_index=True, right_on='index')
    df_n.set_index('index', inplace=True)

    # 변화율 계산하여 새로운 칼럼 추가
    df_n['변화율'] = ((df_n['전력사용량'] - df_n['전력중앙값']) / df_n['전력중앙값']) * 100

    df_n = df_n.dropna()

    # 급증 기준: 동일 요일 동 시간대 4개의 중앙값 대비 증가율 30% 초과
    df_n['급증'] = df_n['변화율'] > 30

    # 전역 변수로 할당
    globals()[f'df_{n}'] = df_n

# 빌딩 1부터 60까지의 급증 갯수 계산
surge = []
for i in range(1, 61):
    building(i)
    df_n = globals()[f'df_{i}']
    surge_count = df_n['급증'].sum()
    surge.append({'건물번호': i, '급증갯수': surge_count})

# 결과를 데이터프레임으로 변환
surge_df = pd.DataFrame(surge)
surge_df.sort_values(by='급증갯수', ascending=False)

df_4.columns

# 데이터 전처리: 학습에 필요 없는 컬럼 제거
# 예를 들어, df_4가 아닌 df 사용
df = df_4.drop(['건물번호','전력사용량', '비전기냉방설비운영', '태양광보유', '월', '일', '요일', '시','일조', '전력중앙값', '변화율'], axis=1)

# Prophet 형식에 맞게 데이터 변환
df = df.rename(columns={'날짜': 'ds', '급증': 'y'})




# 학습 및 테스트 데이터셋 분할
# 마지막 이틀을 테스트 셋으로 사용
test_df = df[df['ds'] >= (df['ds'].max() - pd.Timedelta(days=2))]
train_df = df[df['ds'] < (df['ds'].max() - pd.Timedelta(days=2))]

# Prophet 모델 학습
# model = Prophet(yearly_seasonality=True, daily_seasonality=True)
# model.fit(train_df[['ds', 'y']])



# 모든 설명 변수로 Prophet 모델 학습
model2 = Prophet(yearly_seasonality=True, daily_seasonality=True)

# 추가 회귀 변수 등록
regressors = df.drop(['ds', 'y'], axis=1).columns  # 추가 회귀 변수로 사용할 컬럼들
for reg in regressors:
    model2.add_regressor(reg)

# 모델 학습 (추가 회귀 변수를 함께 사용)
model2.fit(train_df[['ds', 'y'] + list(regressors)])

# 미래 날짜 생성 및 예측
future = model2.make_future_dataframe(periods=len(test_df), freq='H')


# test_df에 해당하는 추가 회귀 변수의 행만 할당
future.loc[future.shape[0] - len(test_df):, regressors] = test_df[regressors].values
# 추가 회귀 변수를 future 데이터프레임에 추가
# future[regressors] = test_df[regressors].values
# future = pd.concat([future, test_df[regressors].reset_index(drop=True)], axis=1)

# 결측값이 있는 열을 선형 보간법으로 채움
# future = future.interpolate(method='linear')

# Prophet 예측 수행
forecast = model2.predict(future)
