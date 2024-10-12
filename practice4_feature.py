# 모델링
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

np.random.seed(42)

# 데이터 불러오기
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
# df.drop('시', axis=1, inplace=True)  # '시' 칼럼 제거
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말: 1, 평일: 0

# # 3개 행씩 묶어 평균 계산
# df_avg = df.groupby(df.index // 3).mean(numeric_only=True)
# df_avg['날짜'] = df.groupby(df.index // 3)['날짜'].first().values



df.columns
df['시'].unique()



# 파생변수 추가
df.columns
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말: 1, 평일: 0
df['기온_1시간전'] = df['기온'].shift(1)  # 1시간 전 기온
df['기온_24시간전'] = df['기온'].shift(24)  # 24시간 전 기온
df['기온_3시간평균'] = df['기온'].rolling(window=3).mean()  # 최근 3시간 평균
df['기온_24시간평균'] = df['기온'].rolling(window=24).mean()  # 최근 24시간 평균
df['풍속_24시간평균'] = df['풍속'].rolling(window=24).mean()  # 최근 24시간 평균
# df['일조_24시간평균'] = df['일조'].rolling(window=24).mean()  # 최근 24시간 평균
df['강수량_24시간평균'] = df['강수량'].rolling(window=24).mean()  # 최근 24시간 평균
# 체감온도 변수 생성
df["체감온도"]=13.12 + 0.6215 *df["기온"] -11.37*(df["풍속"]**0.16) + 0.3965*df["기온"]*(df["풍속"]**0.16)
# # 불쾌지수 변수 생성
df["불쾌지수"]=0.81 *df["기온"]+ 0.01*df["습도"] *(0.99*df["기온"]-14.3)+46.3

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
surge_df.sort_values(by='급증갯수',ascending=False)

df_26[(((df['월']==8) & (df['일']==5))|((df['월']==7) & ((df['일']==29)|(df['일']==22)|(df['일']==15)|(df['일']==8)))) & (df['시'] ==9)]
df_26[(((df['월']==8) & (df['일']==5))) & (df['시'] ==9)][['날짜','전력사용량','전력중앙값','변화율','급증']]

df_1[(df['월']==8) & (df['일']>=22)].drop(columns=['시간_sin','시간_cos','주말','전력중앙값','변화율','년','월','일','시','요일']).head()



print("건물 4, 8월 19일~20일 급증 개수 :",sum(df_59[(df_59['월']==8) & ((df_59['일']==19)|(df_59['일']==20))]['급증'])\
    , "\n건물 59, 8월 20일~21일 급증 개수 :",sum(df_59[(df_59['월']==8) & ((df_59['일']==20)|(df_59['일']==21))]['급증'])\
    , "\n건물 59, 8월 21일~22일 급증 개수 :",sum(df_59[(df_59['월']==8) & ((df_59['일']==21)|(df_59['일']==22))]['급증'])\
    ,"\n건물 59, 8월 22일~23일 급증 개수 :",sum(df_59[(df_59['월']==8) & ((df_59['일']==22)|(df_59['일']==23))]['급증'])\
      ,"\n건물 59, 8월 23일~24일 급증 개수 :",sum(df_59[(df_59['월']==8) & (df_59['일']>=23)]['급증']))


print("건물 59, 8월 19일~20일 급증 개수 :",sum(df_4[(df_4['월']==8) & ((df_4['일']==19)|(df_4['일']==20))]['급증'])\
    , "\n건물 4, 8월 20일~21일 급증 개수 :",sum(df_4[(df_4['월']==8) & ((df_4['일']==20)|(df_4['일']==21))]['급증'])\
    , "\n건물 4, 8월 21일~22일 급증 개수 :",sum(df_4[(df_4['월']==8) & ((df_4['일']==21)|(df_4['일']==22))]['급증'])\
    ,"\n건물 4, 8월 22일~23일 급증 개수 :",sum(df_4[(df_4['월']==8) & ((df_4['일']==22)|(df_4['일']==23))]['급증'])\
      ,"\n건물 4, 8월 23일~24일 급증 개수 :",sum(df_4[(df_4['월']==8) & (df_4['일']>=23)]['급증']))


print("건물 30, 8월 19일~20일 급증 개수 :",sum(df_30[(df_30['월']==8) & ((df_30['일']==19)|(df_30['일']==20))]['급증'])\
    , "\n건물 30, 8월 20일~21일 급증 개수 :",sum(df_30[(df_30['월']==8) & ((df_30['일']==20)|(df_30['일']==21))]['급증'])\
    , "\n건물 30, 8월 21일~22일 급증 개수 :",sum(df_30[(df_30['월']==8) & ((df_30['일']==21)|(df_30['일']==22))]['급증'])\
    ,"\n건물 30, 8월 22일~23일 급증 개수 :",sum(df_30[(df_30['월']==8) & ((df_30['일']==22)|(df_30['일']==23))]['급증'])\
      ,"\n건물 30, 8월 23일~230일 급증 개수 :",sum(df_30[(df_30['월']==8) & (df_30['일']>=23)]['급증']))

print("건물 30, 8월 6일~7일 급증 개수 :",sum(df_30[(df_30['월']==8) & ((df_30['일']==6)|(df_30['일']==7))]['급증'])\
    , "\n건물 30, 8월 7일~8일 급증 개수 :",sum(df_30[(df_30['월']==8) & ((df_30['일']==7)|(df_30['일']==8))]['급증']))

k = df_30[(df_30['월']==8) & ((df_30['일']==7)|(df_30['일']==8))]














np.random.seed(42)
tf.random.set_seed(42)

# 데이터 전처리: 학습에 필요 없는 컬럼 제거
df = df_4.drop(['날짜', '일조','시','건물번호', '전력사용량', '비전기냉방설비운영', '태양광보유', '년', '전력중앙값', '변화율'], axis=1)
len(df_4)
np.median([3698.46, 3902.58, 4106.70, 4310.82])
np.median([3902.58, 4106.70, 4310.82,4514.94])

# polynomialfeatures
poly_features = df[['기온', '풍속', '습도']]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_transformed = poly.fit_transform(poly_features)
poly_columns = poly.get_feature_names_out(['기온', '풍속', '습도'])
poly_df = pd.DataFrame(poly_transformed, columns=poly_columns)
poly_df.drop(['기온', '풍속', '습도'], axis=1, inplace=True)
df = pd.concat([df, poly_df], axis=1)



# 다항 만들기 polynomial features
poly_features = df.drop(columns='급증')
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias =True)
poly_transformed = poly.fit_transform(poly_features)
poly_df_x = pd.DataFrame(poly_transformed, columns = poly.get_feature_names_out(input_features=poly_features.columns))

poly_df_x.columns
len(poly_df_x)
if '급증' in poly_df_x.columns:
    print("급증 컬럼 있다")
else:
    print("없다")
len(poly_df_x['급증'])

# 다항 데이터에, Y값 결합. (서로 인덱스 다름)
poly_df = pd.concat([poly_df_x, df_4[['급증']].reset_index(drop = True)], axis=1)

df.columns




# IV 값을 계산하는 함수
def calculate_iv(df, feature, target):
    # 수치컬럼을 10개의 구간으로 나누기, 중복된 경계값 제거
    try:
        bins = pd.qcut(df[feature], q=10, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Error for feature {feature}: {e}")
        return np.nan  # 에러 발생 시 NaN 반환

    df['구간'] = bins

    # 각 구간의 좋은 사건과 나쁜 사건의 수 계산
    grouped = df.groupby('구간')[target].agg(['count', 'sum'])
    grouped.columns = ['총건수', '긍정건수']
    grouped['부정건수'] = grouped['총건수'] - grouped['긍정건수']

    # 전체 긍정건수와 부정건수 계산
    total_positive = grouped['긍정건수'].sum()
    total_negative = grouped['부정건수'].sum()

    # 각 구간의 비율 계산
    grouped['긍정비율'] = grouped['긍정건수'] / total_positive
    grouped['부정비율'] = grouped['부정건수'] / total_negative

    # IV 값 계산
    grouped['IV'] = (grouped['긍정비율'] - grouped['부정비율']) * np.log(grouped['긍정비율'] / grouped['부정비율'].replace(0, np.nan))
    iv_value = grouped['IV'].sum()

    return iv_value



df_4


# 각 수치형 컬럼의 IV 값 계산
iv_results = {}
find = ['급증','1']
for column in poly_df.columns[np.where(~poly_df.columns.isin(find))[0]]:  # 타겟 컬럼 제외
    iv = calculate_iv(df, column, '급증')
    iv_results[column] = iv

# IV 결과를 데이터프레임으로 변환 및 정렬
iv_df = pd.DataFrame(list(iv_results.items()), columns=['컬럼명', 'IV 값'])
iv_df = iv_df.sort_values(by='IV 값', ascending=False)

# 특정 IV 값을 기준으로 중요한 컬럼 선정 (예: IV > 0.1)
important_columns = iv_df[iv_df['IV 값'] > 0.1]


print(iv_df)
print("중요한 컬럼:")
print(important_columns)



# 각 수치형 컬럼의 IV 값 계산
iv_results = {}
find = ['급증','1']
for column in poly_df.columns[np.where(~poly_df.columns.isin(find))[0]]:  # 타겟 컬럼 제외
    iv = calculate_iv(poly_df, column, '급증')
    iv_results[column] = iv

# IV 결과를 데이터프레임으로 변환 및 정렬
iv_df = pd.DataFrame(list(iv_results.items()), columns=['컬럼명', 'IV 값'])
iv_df = iv_df.sort_values(by='IV 값', ascending=False)

# 특정 IV 값을 기준으로 중요한 컬럼 선정 (예: IV > 0.1)
important_columns = iv_df[iv_df['IV 값'] > 0.1]


print(iv_df)
print("중요한 컬럼:", important_columns['컬럼명'].values )



im_col = important_columns['컬럼명'].values






# IV 계산 함수

# IV 계산 함수
def calculate_iv(df, feature, target):
    lst = []
    total = df[target].count()
    
    for val in df[feature].unique():
        sub_df = df[df[feature] == val]
        good = sub_df[target].sum()
        bad = sub_df[target].count() - good
        good_dist = good / total if total > 0 else 0
        bad_dist = bad / total if total > 0 else 0
        if good_dist > 0 and bad_dist > 0:  # 0으로 나누는 오류 방지
            iv = (good_dist - bad_dist) * np.log(good_dist / bad_dist)
            lst.append({'Value': val, 'Good': good, 'Bad': bad, 'IV': iv})

    if not lst:  # lst가 비어있다면
        return pd.DataFrame(columns=['Value', 'Good', 'Bad', 'IV'])  # 빈 DataFrame 반환

    return pd.DataFrame(lst)


poly_df.columns
len(poly_df.columns)
poly_df.describe()  # poly_df에서 '급증', '1' 제외하기

poly_df.iloc[:,5].describe()



# 기온 구간화
cut_df['기온_구간'] = pd.cut(poly_df['기온'], bins=[15.3, 21.9, 24.8, 28.2, 36.3], labels=['차가움', '보통', '더움', '아주 더움'])

# 풍속 구간화
cut_df['풍속_구간'] = pd.cut(poly_df['풍속'], bins=[-1, 1.1, 1.9, 2.8, 8.3], labels=['저풍속', '중저풍속', '중풍속', '고풍속'])

# 습도 구간화
cut_df['습도_구간'] = pd.cut(poly_df['습도'], bins=[-1, 58, 74, 89, 100], labels=['저습도', '중저습도', '중습도', '고습도'])
# 강수량 구간화
cut_df['강수량_구간'] = pd.cut(poly_df['강수량'], bins=[0-1, 1, 5, 10, 15, 27.5], labels=['없음', '저강수', '중강수', '고강수', '폭우'])

# 일조 구간화
cut_df['일조_구간'] = pd.cut(poly_df['일조'], bins=[-1, 0.1, 0.4, 1], labels=['저일조', '중일조', '고일조'])

cut_df = pd.concat([cut_df, df_4['급증'].reset_index(drop=True)], axis=1)

# 수치형 컬럼을 10개 구간으로 나누기
cut_df = pd.DataFrame({})
target_columns = ['월','일','요일','급증']
for col in poly_df.columns[np.where(~poly_df.columns.isin(target_columns))[0]]:
    # 구간을 생성
    bins = pd.qcut(poly_df[col], q=10, duplicates='drop', retbins=True)
    labels = [f'구간_{i+1}' for i in range(len(bins[1]) - 1)]  # labels 수를 bin edges 수에 맞춤
    cut_df[f'{col}_구간'] = bins[0].cat.rename_categories(labels)  # 카테고리 이름 변경



# 수치형 컬럼을 10개 구간으로 나누기
cut_df = pd.DataFrame({})
target_columns = ['기온','습도']
for col in poly_df.columns[np.where(poly_df.columns.isin(target_columns))[0]]:
    # 구간을 생성
    bins = pd.qcut(poly_df[col], q=10, duplicates='drop', retbins=True)
    labels = [f'구간_{i+1}' for i in range(len(bins[1]) - 1)]  # labels 수를 bin edges 수에 맞춤
    cut_df[f'{col}_구간'] = bins[0].cat.rename_categories(labels)  # 카테고리 이름 변경


col='기온'
bins = pd.qcut(df[col], q=10, duplicates='drop', retbins=True)
labels = [f'구간_{i+1}' for i in range(len(bins[1]) - 1)]  # labels 수를 bin edges 수에 맞춤
cut_df[f'{col}_구간'] = bins[0].cat.rename_categories(labels)  # 카테고리 이름 변경


# 각 설명변수에 대해 IV 계산
iv_results = {}
for feature in cut_df.columns:
    iv_df = calculate_iv(cut_df, feature, '급증')
    iv_value = iv_df['IV'].sum()
    iv_results[feature] = iv_value


# IV 결과 출력
iv_results_df = pd.DataFrame(list(iv_results.items()), columns=['Feature', 'IV'])
print(iv_results_df)

# 중요한 변수 선택
important_features = iv_results_df[iv_results_df['IV'] > 0.1]['Feature']
print("기존 X컬럼 수 : ",len(df.columns[:-1]), "\n 중요변수 수 :", \
      len(important_features),"\n 중요한 변수:", important_features.tolist() \
        , "\n 안 중요한 변수 :", [i for i in df.columns[:-1] if i not in important_features.tolist()])





# 수치형 컬럼을 구간화하는 함수 (10개 구간으로 변경)
def discretize_and_calculate_iv(df, numeric_cols, target):
    iv_results = {}
    
    for col in numeric_cols:
        # 10개 구간으로 구간화
        df[f'{col}_구간'] = pd.qcut(df[col], q=10, duplicates='drop')
        
        # IV 계산
        iv_df = calculate_iv(df, f'{col}_구간', target)
        iv_value = iv_df['IV'].sum()
        iv_results[col] = iv_value
        
        # 구간화된 데이터프레임 출력 (선택 사항)
        print(f"{col} 구간화 및 IV 결과:\n", iv_df)
        
    return iv_results



# 수치형 컬럼 리스트
numeric_columns = poly_df.columns[np.where(~poly_df.columns.isin(target_columns))[0]]

# IV 계산
iv_results = discretize_and_calculate_iv(poly_df, numeric_columns, '급증')
print("IV 결과:", iv_results)




# 학습 및 테스트 데이터셋 분할
test_df = poly_df[(poly_df['월'] == 8) & (poly_df['일'] >= 22)]
train_df = poly_df.drop(test_df.index)

# 특징 변수와 타겟 변수 분리
X_train = train_df.drop('급증', axis=1)
y_train = train_df['급증']
X_test = test_df.drop('급증', axis=1)
y_test = test_df['급증']





im_df = poly_df[im_col]
im_df = pd.concat([im_df, poly_df['월'], poly_df['급증']], axis=1)



# 학습 및 테스트 데이터셋 분할
test_df = im_df[(im_df['월'] == 8) & (im_df['일'] >= 22)]
train_df = im_df.drop(test_df.index)

# 특징 변수와 타겟 변수 분리
X_train = train_df.drop('급증', axis=1)
y_train = train_df['급증']
X_test = test_df.drop('급증', axis=1)
y_test = test_df['급증']




# 1. Lasso 회귀 모델 학습
lasso = Lasso(alpha=0.1)  # alpha 값 조정 필요
lasso.fit(X_train, y_train)

# 2. 중요 변수 선택
selected_features = X_train.columns[lasso.coef_ != 0]
# ['기온 일', '기온 기온_24시간평균', '기온 불쾌지수', '풍속 습도', '습도^2', '습도 강수량', '습도 월',
       '습도 일', '습도 요일', '습도 시간_cos', '습도 기온_24시간전', '습도 기온_3시간평균',
       '습도 기온_24시간평균', '습도 풍속_24시간평균', '습도 강수량_24시간평균', '습도 체감온도', '습도 불쾌지수',
       '강수량 기온_24시간전', '일조 불쾌지수', '일^2', '일 요일', '일 기온_24시간평균', '일 강수량_24시간평균',
       '일 불쾌지수', '요일 기온_24시간전', '요일 기온_24시간평균', '요일 불쾌지수', '시간_sin 불쾌지수',
       '기온_1시간전^2', '기온_1시간전 기온_24시간전', '기온_1시간전 기온_24시간평균', '기온_24시간전^2',
       '기온_24시간전 기온_24시간평균', '기온_3시간평균 체감온도', '기온_3시간평균 불쾌지수',
       '기온_24시간평균 체감온도', '기온_24시간평균 불쾌지수', '일조_24시간평균 불쾌지수', '불쾌지수^2']



# Random Forest 모델 생성 및 학습
rf_clf = RandomForestClassifier(class_weight='balanced',random_state=42)
rf_clf.fit(X_train, y_train)

# 예측 및 평가
prob_y = rf_clf.predict_proba(X_test)[:, 1]  # 양성 클래스의 확률

from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, precision_score, recall_score
from sklearn.metrics import roc_auc_score
# 임계값 목록 생성
thresholds = np.arange(0, 1.1, 0.1)
results = []

roc_auc = roc_auc_score(y_test, prob_y)
for threshold in thresholds:
    pred_y_threshold = (prob_y >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_y_threshold).ravel()
    
    precision = precision_score(y_test, pred_y_threshold)
    recall = recall_score(y_test, pred_y_threshold)
    f1 = f1_score(y_test, pred_y_threshold)  # F1 Score 계산
    fpr = fp / (fp + tn)
    
    # 결과 저장
    results.append({
        'Threshold': threshold,
        'Predicted Positive N': tp,
        'Actual Positive N': tp + fn,
        'Predicted Negative N': tn,
        'Actual Negative N': tn + fp,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'FPR': fpr,
        'ROC AUC': roc_auc 
    })

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)
results_df
# =============================================================

# 임계값 선정
threshold = 0.1
pred_y = (prob_y >= threshold).astype(int)
confusion_matrix(y_test, pred_y)





# 이득도표
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
precision, recall, thresholds_pr = precision_recall_curve(y_test, prob_y)
fpr, tpr, thresholds_roc = roc_curve(y_test, prob_y)

# F1 점수 계산
f1_scores = [f1_score(y_test, prob_y >= thresh) for thresh in thresholds_pr]

# 이득도표 그리기
plt.figure(figsize=(12, 8))

# Precision-Recall 커브
plt.subplot(2, 2, 1)
plt.plot(thresholds_pr, precision[:-1], label="Precision", color="b")
plt.plot(thresholds_pr, recall[:-1], label="Recall", color="g")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Curve")
plt.legend()

# F1 Score 커브
plt.subplot(2, 2, 2)
plt.plot(thresholds_pr, f1_scores, label="F1 Score", color="r")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Threshold")

# ROC 커브 (TPR, FPR)
plt.subplot(2, 2, 3)
plt.plot(fpr, tpr, label="ROC Curve", color="purple")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# 이득도표
gains = np.cumsum(tpr - fpr)
plt.subplot(2, 2, 4)
plt.plot(thresholds_roc, gains, label="Gain", color="brown")
plt.xlabel("Threshold")
plt.ylabel("Cumulative Gain")
plt.title("Gain Chart")
plt.legend()

plt.tight_layout()
plt.show()









catboost_clf = CatBoostClassifier(iterations = 8000, learning_rate=0.1, depth=6, eval_metric='AUC', random_seed=42, verbose=0) #0.671053
catboost_clf.fit(X_train, y_train)

# 예측 및 평가
prob_y = catboost_clf.predict_proba(X_test)[:, 1]  # 양성 클래스의 확률

# 임계값 목록 생성
thresholds = np.arange(0, 1.1, 0.1)
results = []

roc_auc = roc_auc_score(y_test, prob_y)
for threshold in thresholds:
    pred_y_threshold = (prob_y >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_y_threshold).ravel()
    
    precision = precision_score(y_test, pred_y_threshold)
    recall = recall_score(y_test, pred_y_threshold)
    f1 = f1_score(y_test, pred_y_threshold)  # F1 Score 계산
    fpr = fp / (fp + tn)
    
    # 결과 저장
    results.append({
        'Threshold': threshold,
        'Predicted Positive N': tp,
        'Actual Positive N': tp + fn,
        'Predicted Negative N': tn,
        'Actual Negative N': tn + fp,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'FPR': fpr,
        'ROC AUC': roc_auc 
    })

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)
results_df
# =============================================================

# 임계값 선정
threshold = 0.5
pred_y = (prob_y >= threshold).astype(int)
confusion_matrix(y_test, pred_y)

# 이득도표
precision, recall, thresholds_pr = precision_recall_curve(y_test, prob_y)
fpr, tpr, thresholds_roc = roc_curve(y_test, prob_y)

# F1 점수 계산
f1_scores = [f1_score(y_test, prob_y >= thresh) for thresh in thresholds_pr]
