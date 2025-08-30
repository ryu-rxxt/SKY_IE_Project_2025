import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


'''
==============================================================
데이터 읽기
==============================================================
'''

'''
-----------------------------------------------------------
외부 데이터
-----------------------------------------------------------
'''
data_path = "data"
csv_files = glob.glob(os.path.join(data_path, "*.csv"))


dfs_dict = {}
for file_path in csv_files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    var_name = f"{filename}_df"
    df = pd.read_csv(file_path)
    dfs_dict[var_name] = df
    globals()[var_name] = df
    missing_count = df.isna().sum().sum()
    # print(f"{var_name}: loaded with shape {df.shape}, missing values: {missing_count}")



'''
-----------------------------------------------------------
수요 데이터
-----------------------------------------------------------
'''
conn = sqlite3.connect("data/demand_train.db") 
demand_df = pd.read_sql("SELECT * FROM demand_train", conn)
conn.close()


demand_df['date'] = pd.to_datetime(demand_df['date'])
city_country_map = site_candidates_df[['city', 'country']].drop_duplicates() # 수요 데이터에 country 컬럼 추가
demand_df = demand_df.merge(city_country_map, on='city', how='left')
city_demand_df = (demand_df.groupby(['date', 'city','country'])['demand'].sum().reset_index(name='demand_sum').sort_values(['date', 'city'])) # 일별 도시의 수요합






'''
==============================================================
결측치 처리
==============================================================
'''
'''
------------------------------------------------------------
유가: 주말의 유가는 직전 영업일의 것을 사용한다
------------------------------------------------------------
'''
oil_price_df['date'] = pd.to_datetime(oil_price_df['date'])
full_dates = pd.DataFrame({'date': pd.date_range('2018-01-01', '2024-12-31', freq='D')})
oil_price_df = full_dates.merge(oil_price_df, on='date', how='left')
oil_price_df = oil_price_df.sort_values('date').fillna(method='ffill').reset_index(drop=True) # forward fill 이용하여 대체

'''
------------------------------------------------------------
소비자신뢰지수: CAN 결측 처리
------------------------------------------------------------
'''
consumer_confidence_df
consumer_confidence_df['month'] = pd.to_datetime(consumer_confidence_df['month'])

monthly_avg = consumer_confidence_df.groupby('month')['confidence_index'].mean().reset_index() # CAN은 월별 9개국의 평균으로 대체
monthly_avg['country'] = 'CAN'
monthly_avg = monthly_avg[['month', 'country', 'confidence_index']]
consumer_confidence_df = pd.concat([consumer_confidence_df, monthly_avg], ignore_index=True)
consumer_confidence_df = consumer_confidence_df.sort_values(['month', 'country']).reset_index(drop=True)


'''
------------------------------------------------------------
환율: 결측치 & 주말 대체
------------------------------------------------------------
'''
########## 날짜 컬럼의 대문자 소문자로 통일
currency_df = currency_df.rename(columns={'Date': 'date'}) 
currency_df['date'] = pd.to_datetime(currency_df['date'])

########## 평일의 결측치 대체
missing_dates = set(oil_price_df['date']) - set(currency_df['date']) # 유가(주말결측)와 비교하여 사라진 날짜 찾기 -> 주말 외에 결측인 데이터
missing_dates = sorted(list(missing_dates)) # 2019-05-22, 2024-12-31의 환율 결측

new_currency_data = pd.DataFrame([
    {'date': pd.to_datetime('2019-05-22'),
        'EUR=X': 0.8953,
        'KRW=X': 1192.5,
        'JPY=X': 110.37,
        'GBP=X': 0.7876,
        'CAD=X': 1.3468,
        'AUD=X': 1.4485,
        'BRL=X': 4.0623,
        'ZAR=X': 14.368},
    {'date': pd.to_datetime('2024-12-31'),
        'EUR=X': 0.9657,
        'KRW=X': 1442.33,
        'JPY=X': 157.17,
        'GBP=X': 0.7983,
        'CAD=X': 1.1381,
        'AUD=X': 1.6152,
        'BRL=X': 5.5568,
        'ZAR=X': 18.80065}])

new_currency_data = new_currency_data[currency_df.columns] # 직접 구글링한 데이터
currency_df = pd.concat([currency_df, new_currency_data], ignore_index=True)
currency_df = currency_df.sort_values('date').reset_index(drop=True)

########## 미국: 달러 환율 1(기준)로 추가
currency_df['USD=X'] = 1

########## 주말은 직전 영업일의 것을 사용한다
full_dates = pd.DataFrame({'date': pd.date_range('2018-01-01', '2024-12-31', freq='D')})
currency_df = full_dates.merge(currency_df, on='date', how='left')
currency_df = currency_df.sort_values('date').fillna(method='ffill').reset_index(drop=True) # forward fill로 대체




'''
==============================================================
외부 데이터 merge
==============================================================
'''

'''
------------------------------------------------------------
기본  데이터 merge
------------------------------------------------------------
'''
################## 가능한 데이터셋 구분하고 날짜 datetime
for df in [weather_df,marketing_spend_df,holiday_lookup_df,calendar_df, oil_price_df,currency_df]: #기준이 날짜, 도시, 나라로 구분되지 않는 데이터셋은 제외
    df['date'] = pd.to_datetime(df['date'], errors='coerce')    
consumer_confidence_df['month'] = pd.to_datetime(consumer_confidence_df['month'], errors='coerce')
labour_policy_df['year'] = labour_policy_df['year'].astype(int)
dates     = pd.date_range('2018-01-01', '2024-12-31', freq='D')
countries = city_country_map['country'].unique()
cities    = city_country_map['city'].unique()
dates_df = pd.DataFrame({'date': dates})
set_df = dates_df.merge(city_country_map, how='cross') # country나 city로만 존재하는 경우 둘 다 존재하도록 추가

################## 기본 데이터 merge
set_df = set_df.merge(weather_df, on=['date','country'], how='left')
set_df = set_df.merge(marketing_spend_df, on=['date','country'], how='left')
set_df = set_df.merge(calendar_df, on=['date','country'], how='left')
set_df = set_df.merge(oil_price_df, on='date', how='left')

'''
------------------------------------------------------------
공휴일 merge : 어떤 공휴일인지는 중요하지 않기 때문에 0, 1로 인코딩
------------------------------------------------------------
'''
hol = holiday_lookup_df.copy()
hol['is_holiday'] = 1             # 공휴일에 플래그를 만듬
set_df = (set_df.merge(hol[['date','country','is_holiday']], on=['date','country'], how='left').fillna({'is_holiday': 0})) # 공휴일이면 1, 아니면 0
set_df['is_holiday'] = set_df['is_holiday'].astype(int)


'''
------------------------------------------------------------
환율 merge : 기준을 country에 맞추기 위해 구조 변경
------------------------------------------------------------
'''
cur_long = (currency_df.melt(id_vars='date', var_name='cur_code', value_name='currency_rate'))
cur_long['cur_code'] = cur_long['cur_code'].str.replace('=X','') # cur_code: KRW=X 화폐 단위 ----> country: KOR 나라 이름
cur_map = {
    'EUR':['DEU','FRA'], 'KRW':['KOR'], 'JPY':['JPN'],
    'GBP':['GBR'], 'CAD':['CAN'], 'AUD':['AUS'],
    'BRL':['BRA'], 'ZAR':['ZAF'], 'USD':['USA']}
cur_map_df = pd.DataFrame([(code, c) for code, countries in cur_map.items() for c in countries], columns=['cur_code','country'])
cur_long = cur_long.merge(cur_map_df, on='cur_code', how='left')
set_df = set_df.merge(cur_long[['date','country','currency_rate']],on=['date','country'], how='left') # currency_rate가 나라별 환율

 
'''
------------------------------------------------------------
노동법 merge : 년 단위 -> 일 단위
------------------------------------------------------------
'''
lab = labour_policy_df.copy()
lab['year'] = lab['year'].astype(int)
set_df['year'] = set_df['date'].dt.year
set_df = set_df.merge(lab, on=['year','country'], how='left').drop(columns='year') # 연도가 같으면 같은 값


'''
------------------------------------------------------------
소비자 신뢰지수 merge : 반영에 시차 존재 -> 한 달 lag
------------------------------------------------------------
'''
################## set_df에는 한 달 뒤로 들어감
cc = consumer_confidence_df.copy()
cc['month'] = pd.to_datetime(cc['month']).dt.to_period('M') + 1 
set_df['month'] = set_df['date'].dt.to_period('M')
set_df = set_df.merge(cc[['country', 'month', 'confidence_index']], on=['country', 'month'], how='left')

################## cc의 2018-01이 set_df의 2018-02로 들어가기 때문에 2018-01은 na -> 2018-02와 같도록 대체
feb_vals = (set_df[set_df['month'] == pd.Period('2018-02', 'M')].groupby('country')['confidence_index'].first())
set_df['confidence_index'] = set_df['confidence_index'].fillna(set_df['country'].map(feb_vals)) 
set_df = set_df.drop(columns='month')


set_df = set_df.sort_values(['date','country','city']).reset_index(drop=True)




'''
==============================================================
외생 중요변수 선택
==============================================================
'''
END_DATE = '2022-12-31'
ID_COLS  = ['date','city','country']
TARGET   = 'demand_sum'
TH_CORR  = 0.10
TH_VIF   = 10.0

'''
------------------------------------------------------------
외부 데이터 수치형으로 변환하는 함수(corr, VIF 계산시 필요)
------------------------------------------------------------
'''
def make_X(df):
    num = df.select_dtypes(include=[np.number]).copy()
    cat = df.select_dtypes(include=['object','category']).copy()
    drop_cols = [c for c in ID_COLS + [TARGET] if c in num.columns] # id 제거
    num = num.drop(columns=drop_cols, errors='ignore')
    cat = cat.drop(columns=[c for c in ID_COLS if c in cat.columns], errors='ignore')
    dmy = pd.get_dummies(cat, drop_first=True, dtype=int) # 문자열이나 카테고리는 원핫 인코딩 진행
    X = pd.concat([num, dmy], axis=1)
    return X

'''
------------------------------------------------------------
VIF 임계 이상 제거하는 함수
------------------------------------------------------------
'''
def vif_filter(df, th=10.0):
    cols = df.columns.tolist()
    if len(cols) <= 1:
        return cols
    while True:
        Z = add_constant(df[cols], has_constant='add') # 상수항(절편) 추가
        vifs = pd.Series([variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])], index=cols) # i=0은 constant라 제외
        if vifs.max() <= th: break # 모두 10 이하될 때 까지 VIF가 가장 큰 순서로 제거
        cols.remove(vifs.idxmax())
        if len(cols) <= 1: break
    return cols
    
'''
------------------------------------------------------------
외부 데이터 필터링, 외생 변수 선택
------------------------------------------------------------
'''
################## 데이터 준비
left  = set_df.loc[set_df['date'] <= END_DATE].copy()
right = city_demand_df.loc[city_demand_df['date'] <= END_DATE, ['date','city',TARGET]].copy()
df = left.merge(right, on=['date','city'], how='inner') # 수요데이터가 2022까지있기 때문에 기간 맞춤

X_sel = make_X(df)
X_sel = X_sel.loc[:, X_sel.notna().all()] # 결측/상수 컬럼 제거
X_sel = X_sel.loc[:, X_sel.nunique(dropna=False) > 1]
y = df[TARGET]

################## 상관계수 필터링
corr = X_sel.apply(lambda s: s.corr(y))
keep_corr = corr.index[corr.abs() >= TH_CORR].tolist() 
X_sel = X_sel[keep_corr]

################## VIF 필터링
if X_sel.shape[1] > 1:
    keep_vif = vif_filter(X_sel, TH_VIF) 
else:
    keep_vif = X_sel.columns.tolist()

selected_cols = keep_vif  # 최종 선택 변수 이름들

'''
------------------------------------------------------------
선택된 외생 변수 + 수요 데이터 merge
------------------------------------------------------------
'''
X_all = make_X(set_df)
exog_df = pd.concat([set_df[ID_COLS], X_all[[c for c in selected_cols if c in X_all.columns]]], axis=1) # 최종 선택 변수 데이터셋


################## demand_df를 2024-12-31까지로 확장
sku_map = demand_df[['city','sku','country']].drop_duplicates()   
grid = exog_df.merge(sku_map, on=['city','country'], how='left')    # (date,city) × sku 25개 확장
df = grid.merge(demand_df, on=['date','city','sku','country'], how='left')  # 미래 구간은 demand가 Na

################## 시계열성을 위한 feature 생성
df['time_idx'] = (df['date'] - df['date'].min()).dt.days.astype('int32')
df['month'] = df['date'].dt.month.astype('int16')
df['dow']   = df['date'].dt.dayofweek.astype('int16')

for c in ['country','city','sku']:
    df[c] = df[c].astype('category')

num_cols = df.select_dtypes(include=['number']).columns.tolist()
exog_cols = [c for c in num_cols if c not in ['time_idx','month','dow','demand']] # 숫자형 외생변수만 추출 (문자열/카테고리 제외)






'''
==============================================================
2023, 2024 예측
==============================================================
'''


df = df.copy()
cutoff = df.loc[df['demand'].notna(), 'date'].max()  # 수요 데이터가 존재하는 마지막 날(예: 2022-12-31)
gcols = ['city', 'sku']

outs = []

'''
------------------------------------------------------------
# Ridge 회귀 예측
------------------------------------------------------------
'''
for (g_city, g_sku), g in df.sort_values('date').groupby(gcols):
    g = g.copy()
    
    g['lag_365'] = g['demand'].shift(365)     # 1년 전 lag 생성

    '''
    ------------------------------------------------------------
    훈련 데이터셋 구성, Ridge 학습(alpha 튜닝)
    ------------------------------------------------------------
    '''
    tr = g[g['date'] <= cutoff].dropna(subset=['demand', 'lag_365'])

    Xtr = tr[['lag_365'] + exog_cols]      
    ytr = tr['demand']

    alphas = [0.1, 0.5, 1.0, 5.0, 10.0]      # 가장 mse가 낮은 알파를 선택하여 학습
    n_splits = 10
    tscv = TimeSeriesSplit(n_splits=n_splits)

    gs = GridSearchCV(
        estimator=Ridge(fit_intercept=True),
        param_grid={'alpha': alphas},
        scoring='neg_mean_squared_error',
        cv=tscv
    )
    gs.fit(Xtr, ytr)
    model = gs.best_estimator_

    g['pred'] = np.nan  # pred로 예측값 저장 

    '''
    ------------------------------------------------------------
    2023 예측 (2018-2022 데이터 기반)
    ------------------------------------------------------------
    '''
    mask23 = (g['date'] > cutoff) & (g['date'].dt.year == 2023)
    if mask23.any():
        X23 = g.loc[mask23, ['lag_365'] + exog_cols].copy()
        X23['lag_365'] = X23['lag_365'].ffill().bfill().fillna(Xtr['lag_365'].mean())
        y23 = model.predict(X23)
        g.loc[mask23, 'pred'] = np.clip(y23, 0, None)   # 0이하면 0으로 clip
        g.loc[mask23, 'demand'] = g.loc[mask23, 'pred'] # 2024 예측에 사용하기 위해 demand에 채움

    '''
    ------------------------------------------------------------
    2024 예측 (2018-2023 데이터 기반)
    ------------------------------------------------------------
    '''
    g['lag_365'] = g['demand'].shift(365)
    mask24 = (g['date'] > cutoff) & (g['date'].dt.year == 2024)
    if mask24.any():
        X24 = g.loc[mask24, ['lag_365'] + exog_cols].copy()
        X24['lag_365'] = X24['lag_365'].ffill().bfill().fillna(Xtr['lag_365'].mean())
        y24 = model.predict(X24)
        g.loc[mask24, 'pred'] = np.clip(y24, 0, None)

    fu = g[g['date'] > cutoff][['date', 'sku', 'city', 'pred']].rename(columns={'pred': 'demand'})
    outs.append(fu)

'''
------------------------------------------------------------
최종 결과 결합 및 템플릿 맞춰서 저장
------------------------------------------------------------
'''
out_df = (
    pd.concat(outs, ignore_index=True)
    .sort_values(['city', 'sku', 'date'])
    .reset_index(drop=True))
out_df = out_df.rename(columns={'demand': 'mean'})
out_df['mean'] = out_df['mean'].round().astype(int)
# out_df.to_csv("forecast_submission_template.csv", index=False)
