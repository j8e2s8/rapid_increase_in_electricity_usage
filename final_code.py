import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier




# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project2/data_week2.csv

# 컬럼명 바꾸기
df.columns  # ['num', 'date_time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)','강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유']
df = df.rename(columns = {'num' : '건물번호', 'date_time' : '날짜' , '전력사용량(kWh)' : '전력사용량' , '기온(°C)':'기온', '풍속(m/s)' :'풍속'  , '습도(%)':'습도' , '강수량(mm)':'강수량', '일조(hr)' : '일조'  })
df.head()
df.info()



# 범주컬럼의 범주별로 시계열 그래프 그리기
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


timeline(df, '전력사용량')
timeline(df, '기온')
timeline(df, '풍속')
timeline(df, '습도')
timeline(df, '강수량')
timeline(df, '일조')


# 데이터 정보 확인
df.describe()


# 컬럼별로 확률분포 그려보기
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


kde(df)


# 급증 정의 _ 8월3일~8월9일 전력사용량 그래프
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


# 8월 5일 9시 데이터 확인
df[(df['월']==8) & (df['일']==5) & (df['시'] ==9) & (df['건물번호']==26)]


# 데이터 정보 확인
df.info()

for i in df.columns:
	print(f'{i}컬럼의 unique 개수 :',len(df[i].unique()))
      

cols = ['건물번호','비전기냉방설비운영','태양광보유']
for i in cols:
	print(f'{i}컬럼의 unique :', df[i].unique())



# 데이터 자료형 바꾸기
df['건물번호'] = df['건물번호'].astype('object')
df['날짜'] = pd.to_datetime(df['날짜'])
df['비전기냉방설비운영'] = df['비전기냉방설비운영'].astype('boolean')
df['태양광보유'] = df['태양광보유'].astype('boolean')


# 히트맵
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df[["전력사용량","기온","풍속","습도","강수량","일조"]].corr(),annot=True, fmt=".2f",cmap="Blues")

# 기온과 일조 관계 그래프
sns.lineplot(x=df["기온"],y=df["일조"])

# 풍속 시계열 그래프
plt.figure(figsize=(10,6))
sns.lineplot(x=df.query("월==6 & 일>=1 & 일<8")["날짜"],y=df.query("월==6 & 일>=1 & 일<8")["풍속"])
plt.title("6월 첫째 주의 풍속 그래프",fontsize=16)

# 기온 시계열 그래프
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(10,6))
sns.lineplot(x=df.query("월==6 & 일>=1 & 일<8")["날짜"],y=df.query("월==6 & 일>=1 & 일<8")["기온"])
plt.title("6월 첫째 주의 기온 그래프",fontsize=16)



# 시간 인코딩
df['년'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['일'] = df['날짜'].dt.day
df['요일'] = df['날짜'].dt.dayofweek  # 0: 월요일, 6: 일요일
df['시'] = df['날짜'].dt.hour
df['시간_sin'] = np.sin(2 * np.pi * df['시'] / 24)
df['시간_cos'] = np.cos(2 * np.pi * df['시'] / 24)



# 파생변수 추가
df.columns
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말: 1, 평일: 0
df['기온_1시간전'] = df['기온'].shift(1)  # 1시간 전 기온
df['기온_24시간전'] = df['기온'].shift(24)  # 24시간 전 기온
df['기온_3시간평균'] = df['기온'].rolling(window=3).mean()  # 최근 3시간 평균
df['기온_24시간평균'] = df['기온'].rolling(window=24).mean()  # 최근 24시간 평균
df['풍속_24시간평균'] = df['풍속'].rolling(window=24).mean()  # 최근 24시간 평균
df['일조_24시간평균'] = df['일조'].rolling(window=24).mean()  # 최근 24시간 평균
df['강수량_24시간평균'] = df['강수량'].rolling(window=24).mean()  # 최근 24시간 평균
df["체감온도"]=13.12 + 0.6215 *df["기온"] -11.37*(df["풍속"]**0.16) + 0.3965*df["기온"]*(df["풍속"]**0.16)
df["불쾌지수"]=0.81 *df["기온"]+ 0.01*df["습도"] *(0.99*df["기온"]-14.3)+46.3




# 건물마다 데이터셋 생성
def building(n):
    # 지정된 건물 번호로 필터링
    df_n = df_avg.query(f'건물번호 == {n}')

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

# 결과를 데이터프레임으로 변환 _ 급증 개수가 높은 건물 확인
surge_df = pd.DataFrame(surge)
surge_df.sort_values(by='급증갯수',ascending=False)




# 인천 일기예보 데이터 가져오기
from functools import reduce

city="인천"
location="논현고잔동"
df_prec = pd.read_csv(f"{city}/{location}_강수_202006_202008.csv")
df_temp = pd.read_csv(f"{city}/{location}_기온_202006_202008.csv")
df_humi = pd.read_csv(f"{city}/{location}_습도_202006_202008.csv")
df_wind = pd.read_csv(f"{city}/{location}_풍속_202006_202008.csv")

df_list=[df_prec, df_temp, df_humi, df_wind]
# 칼럼명 변경
df_prec.columns=["일","시간","강수량"]
df_temp.columns=["일","시간","기온"]
df_humi.columns=["일","시간","습도"]
df_wind.columns=["일","시간","풍속"]
# '월' 칼럼 생성 & 전처리리
for i in df_list:
    i["시간"]=i["시간"]/100
    i["월"]=6

a, b=df_prec[(df_prec["일"]==' Start : 20200701 ')|(df_prec["일"]==' Start : 20200801 ')].index

for i in df_list:
    i.loc[a:b,"월"]=7
    i.loc[b:,"월"]=8
    i.drop(i[(i["일"]==' Start : 20200701 ')|(i["일"]==' Start : 20200801 ')].index,inplace=True)
    i.reset_index(drop=True)
    i["일"]=i["일"].astype(int)

df_incheon = reduce(lambda left, right: pd.merge(left, right, how='outer', on=["월", "일", "시간"]), df_list)
df_incheon = df_incheon[["월","일","시간","기온","강수량","습도","풍속"]]

df_incheon.head()



# 서울 일기예보 데이터 가져오기
city="서울"
location="청운효자동"
df_prec = pd.read_csv(f"{city}/{location}_강수_202006_202008.csv")
df_temp = pd.read_csv(f"{city}/{location}_기온_202006_202008.csv")
df_humi = pd.read_csv(f"{city}/{location}_습도_202006_202008.csv")
df_wind = pd.read_csv(f"{city}/{location}_풍속_202006_202008.csv")

df_list=[df_prec, df_temp, df_humi, df_wind]
# 칼럼명 변경
df_prec.columns=["일","시간","강수량"]
df_temp.columns=["일","시간","기온"]
df_humi.columns=["일","시간","습도"]
df_wind.columns=["일","시간","풍속"]

# '월' 칼럼 생성 & 전처리리
for i in df_list:
    i["시간"]=i["시간"]/100
    i["월"]=6

a, b=df_prec[(df_prec["일"]==' Start : 20200701 ')|(df_prec["일"]==' Start : 20200801 ')].index

for i in df_list:
    i.loc[a:b,"월"]=7
    i.loc[b:,"월"]=8
    i.drop(i[(i["일"]==' Start : 20200701 ')|(i["일"]==' Start : 20200801 ')].index,inplace=True)
    i.reset_index(drop=True)
    i["일"]=i["일"].astype(int)

df_seoul = reduce(lambda left, right: pd.merge(left, right, how='outer', on=["월", "일", "시간"]), df_list)
df_seoul = df_seoul[["월","일","시간","기온","강수량","습도","풍속","강수형태"]]
df_seoul.head()






# ----------------------------------------------
# cat boost 최종모델 _ 건물 n -> df_n 으로 설정해서 돌리기

# # df_30에서 급증 갯수가 3개 이상인 날짜 필터링
# df_true_counts = df_30[df_30['급증']].groupby(df_30['날짜'].dt.date).size()
# days_with_three_or_more_true = df_true_counts[df_true_counts >= 3]

# 데이터 전처리: 학습에 필요 없는 컬럼 제거
## 건물_n 데이터에 대해서 df_n으로 설정해주기
df = df_4.drop(['날짜', '시', '일조', '건물번호', '전력사용량', '비전기냉방설비운영', '태양광보유', '년', '날짜', '전력중앙값', '변화율'], axis=1)
df_plot=df_4  # 모델을 돌린 후, 시각화 돌릴 때 사용할 건물_n 데이터 지정해주기




# 학습 및 테스트 데이터셋 분할
# 6, 7일
# test_df = df[(df['월'] == 8) & ((df['일'] == 6)|(df['일'] == 7))]
# train_df = df[(df['월'] < 8) | ((df['월'] == 8) & (df['일'] < 6))]

# 7, 8
# test_df = df[(df['월'] == 8) & ((df['일'] == 7)|(df['일'] == 8))]
# train_df = df[(df['월'] < 8) | ((df['월'] == 8) & (df['일'] < 7))]

# 22, 23일
test_df = df[(df['월'] == 8) & ((df['일'] == 22)|(df['일'] == 23))]
train_df = df[(df['월'] < 8) | ((df['월'] == 8) & (df['일'] < 22))]

# 23, 24일
# test_df = df[(df['월'] == 8) & (df['일'] >= 23)]
# train_df = df.drop(test_df.index)




np.random.seed(42)
# 특징 변수와 타겟 변수 분리
X_train = train_df.drop('급증', axis=1).values
y_train = train_df['급증'].values
X_test = test_df.drop('급증', axis=1).values
y_test = test_df['급증'].values

# 그리드서치
# TimeSeriesSplit 객체 생성
tscv = TimeSeriesSplit(n_splits=5)
# Grid Search에 사용할 파라미터 범위 정의
param_grid = {
    'iterations': [1000,2000,3000,4000,5000]
}

scale_pos_weight = (len(y_test) - y_test.sum()) / y_test.sum()
catboost_clf = CatBoostClassifier(scale_pos_weight=scale_pos_weight, learning_rate=0.1, depth=6, eval_metric='AUC', random_seed=42, verbose=0)

grid_search = GridSearchCV(estimator=catboost_clf, param_grid=param_grid, 
                           scoring='roc_auc', cv=tscv, n_jobs=-1)

grid_search.fit(X_train, y_train)

grid_search.best_params_
catboost_clf = grid_search.best_estimator_

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
threshold = 0.3
pred_y = (prob_y >= threshold).astype(int)
confusion_matrix(y_test, pred_y)


## 그래프=============================

pred_df = pd.DataFrame({
    '날짜': df_plot[(df_plot['월'] == 8) & ((df_plot['일'] == 19)|(df_plot['일'] == 20))]['날짜'],
    '예측급증': pred_y,
    '실제급증': df_plot[(df_plot['월'] == 20) & ((df_plot['일'] == 19)|(df_plot['일'] == 20))]['급증']
})



time_df = df_plot[(df_plot['월'] == 8) & ((df_plot['일'] == 19)|(df_plot['일'] == 20))]
highlight = pred_df['예측급증']
highlight2 = time_df['급증']
plt.figure(figsize=(10,7))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=time_df , x='날짜', y='전력사용량', color='black')
#sns.scatterplot(data=time_df , x='날짜', y='전력사용량' ,color='black')
plt.fill_between(pred_df['날짜'], time_df['전력사용량'],where=highlight, color='red', alpha=0.5, label='급증 구간')
plt.fill_between(pred_df['날짜'], time_df['전력사용량'],where=highlight2, color='blue', alpha=0.5, label='급증 구간')
monthly_ticks = pd.date_range(start=pred_df['날짜'].min(), end=pred_df['날짜'].max(), freq='D')
# 날짜와 요일을 함께 표시
tick_labels = [f"{date.strftime('%Y-%m-%d')}\n{date.strftime('%a')}" for date in monthly_ticks]
plt.xticks(monthly_ticks, tick_labels)

##

# 이득도표
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


# 피처 중요도 가져오기
feature_importances = catboost_clf.get_feature_importance()

# 피처 이름
feature_names = train_df.drop('급증', axis=1).columns

# 변수 중요도 데이터프레임 생성
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# 중요도 기준으로 내림차순 정렬
importance_df = importance_df.sort_values(by='Importance')

# 피처 중요도 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

# 정렬된 변수 중요도 데이터프레임 출력
importance_df

importance_df['Feature'].values.shape

















