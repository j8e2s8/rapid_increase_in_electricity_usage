import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from catboost import CatBoostClassifier

np.random.seed(42)

## 필요한 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project2/data_week2.csv

# 컬럼명 바꾸기
df = df.rename(columns = {'num' : '건물번호', 'date_time' : '날짜' , '전력사용량(kWh)' : '전력사용량' , '기온(°C)':'기온', '풍속(m/s)' :'풍속'  , '습도(%)':'습도' , '강수량(mm)':'강수량', '일조(hr)' : '일조'  })
len(df.query("전력사용량==0"))

# 데이터 자료형 바꾸기
df['날짜'] = pd.to_datetime(df['날짜'])
df['비전기냉방설비운영'] = df['비전기냉방설비운영'].astype('boolean')
df['태양광보유'] = df['태양광보유'].astype('boolean')

# 인코딩
df['년'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek  # 0: 월요일, 6: 일요일
df['시'] = df['날짜'].dt.hour
df['시간_sin'] = np.sin(2 * np.pi * df['시'] / 24)
df['시간_cos'] = np.cos(2 * np.pi * df['시'] / 24)
df.drop('시', axis=1, inplace=True)  # '시' 칼럼 제거
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말: 1, 평일: 0

# 3개 행씩 묶어 평균 계산
df_avg = df.groupby(df.index // 3).mean(numeric_only=True)
df_avg['날짜'] = df.groupby(df.index // 3)['날짜'].first().values
    
# 건물 번호에 따른 데이터 반환 함수
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

    df_n = df_n.dropna()

    # # 정규화
    # transformed_variable, lambda_value = stats.boxcox(df_n['변화율'] + 1)
    # df_n['변화율정규화'] = (transformed_variable-transformed_variable.mean())/transformed_variable.std()

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
surge_df.sort_values(by='급증갯수',ascending=False)

# building(1)
# df_1['급증'].sum()
# df_1['변화율'].describe()

# building(2)
# df_2['급증'].sum()
# df_2['변화율'].describe()

# building(3)
# df_3['급증'].sum()
# df_3['변화율'].describe()

# building(4)
# df_4['급증'].sum()
# df_4['변화율'].describe()

# building(5)
# df_5['급증'].sum()
# df_5['변화율'].describe()

# building(6)
# df_6['급증'].sum()
# df_6['변화율'].describe()

# df_4.groupby(['비전기냉방설비운영','태양광보유']).agg(갯수=('급증','sum')) # 건물 조건별 급증 횟수


# # 그래프 그리기
# plt.figure(figsize=(25, 20))
# plt.plot(df_4['전력사용량'], label='전력 사용량 ', color='blue')

# # 급증 상황 표시
# plt.scatter(df_4.index[df_4['급증']], df_4['전력사용량'][df_4['급증']], 
#              color='red', label='급증 상황', marker='o', s=30)

# # 그래프 그리기
# plt.figure(figsize=(25, 20))
# plt.plot(df_6['전력사용량'], label='전력 사용량 ', color='blue')

# # 급증 상황 표시
# plt.scatter(df_6.index[df_6['급증']], df_6['전력사용량'][df_6['급증']], 
#              color='red', label='급증 상황', marker='o', s=30)

# plt.xlim([3400,3600])


# # 학습 및 테스트 데이터셋 분할
# df_4.columns
# df_4.drop(['건물번호','전력사용량','비전기냉방설비운영','태양광보유','년','날짜','전력중앙값','변화율'], axis=1, inplace=True)  # '시' 칼럼 제거

# test_4 = df_4[(df_4['월'] == 8) & (df_4['일']>=22)]
# train_4 = df_4.drop(test_4.index)

# train_x_4 = train_4.drop('급증', axis=1)
# train_y_4 = train_4['급증']

# test_x_4 = test_4.drop('급증', axis=1)
# test_y_4 = test_4['급증']

# # 모델 생성 및 예측
# xgb_clf = XGBClassifier(random_state=42)
# xgb_clf.fit(train_x_4, train_y_4)

# pred_y_4 = xgb_clf.predict(test_x_4)

# # 성능 평가
# f1_score(test_y_4, pred_y_4)
# confusion_matrix(test_y_4, pred_y_4)
df_4['주말'].sum()

def train_and_evaluate_xgb(df):
    # 데이터 전처리: 학습에 필요 없는 컬럼 제거
    df = df.drop(['날짜','건물번호', '전력사용량', '비전기냉방설비운영', '태양광보유', '년', '날짜', '전력중앙값', '변화율'], axis=1)

    # 학습 및 테스트 데이터셋 분할
    test_df = df[(df['월'] == 8) & (df['일'] >= 17)]
    train_df = df.drop(test_df.index)

    train_x = train_df.drop('급증', axis=1)
    train_y = train_df['급증']
    test_x = test_df.drop('급증', axis=1)
    test_y = test_df['급증']

    # SMOTE 적용
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    train_x, train_y = smote.fit_resample(train_x, train_y)

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
    }

    # GridSearchCV 초기화 및 학습
    xgb_clf = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb_clf, param_grid, scoring='f1', cv=3, n_jobs=-1)
    grid_search.fit(train_x, train_y)

    # 최적 모델로 예측
    best_model = grid_search.best_estimator_
    pred_y = best_model.predict(test_x)

    # # 모델 생성 및 학습
    # scale_pos_weight = train_y.shape[0] / train_y.sum()
    # xgb_clf = XGBClassifier(random_state=42)
    # xgb_clf.fit(train_x, train_y)

    # # 예측
    # pred_y = xgb_clf.predict(test_x)

    # 성능 평가
    f1 = f1_score(test_y, pred_y)
    cm = confusion_matrix(test_y, pred_y)

    print("Confusion Matrix:\n", cm)

train_and_evaluate_xgb(df_4)
train_and_evaluate_xgb(df_30)
train_and_evaluate_xgb(df_19)
train_and_evaluate_xgb(df_59)


df_4[(df_4['월'] == 8) & (df_4['일'] >= 17)]['급증'].sum() # 건물 4 -> 12 급증
df_30[(df_30['월'] == 8) & (df_30['일'] >= 17)]['급증'].sum()  # 건물 30 -> 0 급증
df_19[(df_19['월'] == 8) & (df_19['일'] >= 17)]['급증'].sum() # 건물 19 -> 21 급증
df_59[(df_59['월'] == 8) & (df_59['일'] >= 17)]['급증'].sum()  # 건물 59 -> 7 급증

df_4['급증'].sum()



def train_and_evaluate_lstm(df, window_size=3):

    np.random.seed(42)
    tf.random.set_seed(42)

    # 데이터 전처리: 학습에 필요 없는 컬럼 제거
    df = df.drop(['날짜', '건물번호', '전력사용량', '비전기냉방설비운영', '태양광보유', '년', '날짜', '전력중앙값', '변화율'], axis=1)

    # 학습 및 테스트 데이터셋 분할
    test_df = df[(df['월'] == 8) & (df['일'] >= 17)]
    train_df = df.drop(test_df.index)

    # 특징 변수와 타겟 변수 분리
    X_train = train_df.drop('급증', axis=1).values
    y_train = train_df['급증'].values
    X_test = test_df.drop('급증', axis=1).values
    y_test = test_df['급증'].values

    # MinMaxScaler로 정규화
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE 적용
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 데이터를 LSTM에 맞게 3차원으로 변환
    X_train = np.array([X_train[i:i+window_size] for i in range(len(X_train) - window_size)])
    y_train = y_train[window_size:]
    X_test = np.array([X_test[i:i+window_size] for i in range(len(X_test) - window_size)])
    y_test = y_test[window_size:]

    # 모델 정의
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1, activation='sigmoid')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['recall'])

    # # 조기 종료 콜백 설정
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 모델 학습
    model.fit(X_train, y_train, epochs=400, batch_size=14, validation_split=0.2)#, callbacks=[early_stopping])

# (400, 16 , 1/11)

    # 예측 및 평가
    pred_y = (model.predict(X_test) > 0.5).astype(int).flatten()
    f1 = f1_score(y_test, pred_y)
    cm = confusion_matrix(y_test, pred_y)

    print("Confusion Matrix:\n", cm)

train_and_evaluate_lstm(df_4)








def train_and_evaluate_catboost(df):

    # 데이터 전처리: 학습에 필요 없는 컬럼 제거
    df = df.drop(['날짜', '건물번호', '전력사용량', '비전기냉방설비운영', '태양광보유', '년', '전력중앙값', '변화율'], axis=1)

    # 학습 및 테스트 데이터셋 분할
    test_df = df[(df['월'] == 8) & (df['일'] >= 17)]
    train_df = df.drop(test_df.index)

    # 특징 변수와 타겟 변수 분리
    X_train = train_df.drop('급증', axis=1)
    y_train = train_df['급증']
    X_test = test_df.drop('급증', axis=1)
    y_test = test_df['급증']

    # MinMaxScaler로 정규화
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE 적용
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # CatBoostClassifier 초기화
    model = CatBoostClassifier(iterations=1000, 
                               depth=6, 
                               learning_rate=0.1, 
                               loss_function='Logloss', 
                               verbose=100)

    # 모델 훈련
    model.fit(X_train, y_train)

    # 예측 및 평가
    pred_y = model.predict(X_test)

    f1 = f1_score(y_test, pred_y)
    cm = confusion_matrix(y_test, pred_y)

    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)

train_and_evaluate_catboost(df_4)