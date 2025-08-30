def main():
    '''
    ========================================
    1. module import & load raw data
    ========================================
    '''
    # ------------- 기본 import ------------
    import os, glob, argparse, sqlite3
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.seasonal import STL
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    # ---------------- CLI ----------------
    ap = argparse.ArgumentParser(description="SKU Demand Forecast (single-file)")
    ap.add_argument("--data_path",   type=str, required=True, help="외부 CSV들이 있는 폴더 경로")
    ap.add_argument("--sqlite_path", type=str, required=True, help="demand_train.db 경로")
    ap.add_argument("--out_csv",     type=str, default="forecast_submission_template.csv")
    args = ap.parse_args()

    
    '''
    ---------------- loader ----------------
    @ load_csvs_to_dict
        - 입력 
            data_path: path, 제공된 데이터가 모두 포함된 파일의 경로
        - 설명: 폴더의 CSV들을 {filename}_df 형태의 키로 dict 반환
        - 출력 
            dfs: dict, 외부 데이터의 파일명
            
    @ load_demand: 
        - 입력
            sqlite_path: path, 수요 데이터 db의 경로
            site_candidates_df: dataframe, 나라와 도시가 매칭되어있는 데이터
        - 설명: 수요 데이터 불러와서 country 추가, 도시별로 모든 sku의 수요 합 계산
        - 출력
            ddf: dataframe, 나라 컬럼 추가한 수요 데이터
            city_country_map: dataframe, 나라와 도시의 매칭 데이터
            city_demand: dataframe, 도시별 수요 합(demand_sum)을 기준으로 하는 수요 데이터
    -------------------------------------------
    '''
    def load_csvs_to_dict(data_path):
        dfs = {}
        for fp in glob.glob(os.path.join(data_path, "*.csv")):
            name = os.path.splitext(os.path.basename(fp))[0] # csv 형식에서 파일명 추출
            var_name = f"{name}_df"
            df = pd.read_csv(fp)
            dfs[var_name] = df
        return dfs

    def load_demand(sqlite_path, site_candidates_df):
        conn = sqlite3.connect(sqlite_path)
        ddf = pd.read_sql("SELECT * FROM demand_train", conn)
        conn.close()
        ddf['date'] = pd.to_datetime(ddf['date'])
        city_country_map = site_candidates_df[['city','country']].drop_duplicates() # 도시에 따른 국가가 매칭되어있는 site_candidates_df 이용
        ddf = ddf.merge(city_country_map, on='city', how='left') # 수요 데이터에 country 컬럼 추가
        city_demand = (ddf.groupby(['date','city','country'])['demand'] # 도시별 수요 합 계산
                          .sum().reset_index(name='demand_sum')
                          .sort_values(['date','city']))
        return ddf, city_country_map, city_demand

    
    '''
    ========================================
    2. 전처리 후 외부 데이터 merge
    ========================================

    @ prepare_oil: 결측치인 주말의 유가를 전날의 유가로 대체(oil_price_df -> oil)
            
    @ prepare_consumer_confidence: 결측치인 캐나다의 소비자신뢰지수를 모든 나라의 평균으로 대체(cc_df -> cc)
    
    @ prepare_currency: 평일 결측치 2일은 구글링해서 대체, 주말 결측치는 전날의 환율로 대체, 기준이 되는 미국도 USD=X: 1로 추가(currency_df -> cur)
    
    @ merge_base_frames: 날짜가 일 단위인 데이터들 merge(weather_df, marketing_spend_df, calendar_df, oil_df -> set_df)
    
    @ merge_holiday: 공휴일이면 1, 평일이면 0으로 인코딩하여 merge(hol_df -> set_df)
    
    @ merge_currency: 화폐 단위를 나라명으로 매칭(KRW: KOR)하고 수치는 currency_rate로 변환하여 merge(currency_df -> set_df)
    
    @ merge_labour: 년 단위를 일 단위로 바꾸어 merge(labour_df -> set_df)
    
    @ merge_cc_with_lag: 수치 반영에 시간이 소요됨을 고려하여 해당 달의 수치를 다음 달로 추가, 달 단위를 일 단위로 바꾸어 merge(cc_df -> set_df)
    -------------------------------------------
    '''
    def prepare_oil(oil_price_df):
        oil = oil_price_df.copy()
        oil['date'] = pd.to_datetime(oil['date'])
        full = pd.DataFrame({'date': pd.date_range('2018-01-01','2024-12-31',freq='D')}) # 전체 기간 생성
        oil = full.merge(oil, on='date', how='left').sort_values('date').ffill().reset_index(drop=True) # 주말 결측을 전날 값으로 forward fill
        return oil

    def prepare_consumer_confidence(cc_df):
        cc = cc_df.copy()
        cc['month'] = pd.to_datetime(cc['month'])
        monthly_avg = cc.groupby('month')['confidence_index'].mean().reset_index() # 월별 소비자 신뢰지수의 평균 계산
        monthly_avg['country'] = 'CAN' # 평균을 캐나다로 대체
        monthly_avg = monthly_avg[['month','country','confidence_index']]
        cc = pd.concat([cc, monthly_avg], ignore_index=True).sort_values(['month','country']).reset_index(drop=True)
        return cc

    def prepare_currency(currency_df, oil_df):
        cur = currency_df.rename(columns={'Date':'date'}).copy()
        cur['date'] = pd.to_datetime(cur['date'])
        
        missing_dates = sorted(list(set(oil_df['date']) - set(cur['date']))) # 유가와 비교했을 때 결측인 날들(2일)
        if missing_dates:
            new_rows = pd.DataFrame([
                {'date': pd.to_datetime('2019-05-22'),'EUR=X':0.8957,'KRW=X':1192.72,'JPY=X':110.56,'GBP=X':0.7871,'CAD=X':1.3404,'AUD=X':1.4529,'BRL=X':4.0387,'ZAR=X':14.4183},
                {'date': pd.to_datetime('2024-12-31'),'EUR=X':0.9602,'KRW=X':1472.04,'JPY=X':157.0315,'GBP=X':0.7971,'CAD=X':1.4353,'AUD=X':1.6081,'BRL=X':6.1783,'ZAR=X':18.7862}
            ])
            
            new_rows = new_rows[[c for c in cur.columns]] # 직접 검색한 환율로 대체
            cur = pd.concat([cur, new_rows], ignore_index=True)
        cur = cur.sort_values('date').reset_index(drop=True)
        cur['USD=X'] = 1 # USD 환율은 기준이기 때문에 항상 1로 설정
        full = pd.DataFrame({'date': pd.date_range('2018-01-01','2024-12-31',freq='D')}) # 전체 기간 설정 후 주말의 결측치 전날 값으로 forward fill
        cur = full.merge(cur, on='date', how='left').sort_values('date').ffill().reset_index(drop=True)
        return cur

    def merge_base_frames(set_df, weather_df, marketing_spend_df, calendar_df, oil_df):
        for df in [weather_df, marketing_spend_df, calendar_df, oil_df]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce') # 통일 구조인 날짜, 나라, 도시로 잘 정리된 데이터들 merge
        set_df = set_df.merge(weather_df, on=['date','country'], how='left')
        set_df = set_df.merge(marketing_spend_df, on=['date','country'], how='left')
        set_df = set_df.merge(calendar_df, on=['date','country'], how='left')
        set_df = set_df.merge(oil_df, on='date', how='left')
        return set_df

    def merge_holiday(set_df, hol_df):
        hol = hol_df.copy()
        if 'date' in hol.columns:
            hol['date'] = pd.to_datetime(hol['date'], errors='coerce')  
        hol['is_holiday'] = 1 # 공휴일이면 1로 매핑
        set_df = (set_df.merge(hol[['date','country','is_holiday']],
                               on=['date','country'], how='left')
                        .fillna({'is_holiday':0})) # 공휴일이 아니면 0으로 매핑
        set_df['is_holiday'] = set_df['is_holiday'].astype(int) # set_df에 is_holiday로 추가(어떤 공휴일인지는 의미없음)
        return set_df

    def merge_currency(set_df, currency_df):
        cur_long = currency_df.melt(id_vars='date', var_name='cur_code', value_name='currency_rate') # melt로 열과 행 변환
        cur_long['cur_code'] = cur_long['cur_code'].str.replace('=X','', regex=False) # =X 지우고 화폐 단위에서 나라 약어로 매핑(독일과 프랑스는 공통적으로 유로화)
        cur_map = {'EUR':['DEU','FRA'],'KRW':['KOR'],'JPY':['JPN'],'GBP':['GBR'],
                   'CAD':['CAN'],'AUD':['AUS'],'BRL':['BRA'],'ZAR':['ZAF'],'USD':['USA']}
        cur_map_df = pd.DataFrame([(k,v) for k,vs in cur_map.items() for v in vs],
                                  columns=['cur_code','country'])
        cur_long = cur_long.merge(cur_map_df, on='cur_code', how='left') 
        return set_df.merge(cur_long[['date','country','currency_rate']], on=['date','country'], how='left') # 맵에 따라서 set_df에 merge(환율은 currency_rate)

    def merge_labour(set_df, labour_df):
        lab = labour_df.copy()
        lab['year'] = lab['year'].astype(int)
        set_df['year'] = set_df['date'].dt.year # 연 단위를 일 단위로 확장
        set_df = set_df.merge(lab, on=['year','country'], how='left').drop(columns='year')
        return set_df

    def merge_cc_with_lag(set_df, cc_df):
        cc = cc_df.copy()
        cc['month'] = pd.to_datetime(cc['month']).dt.to_period('M') + 1  # 한 달 후 반영
        set_df['month'] = set_df['date'].dt.to_period('M')
        set_df = set_df.merge(cc[['country','month','confidence_index']], on=['country','month'], how='left')
        feb_vals = (set_df[set_df['month'] == pd.Period('2018-02','M')] # 첫 달인 2018-01은 불가피하게 2018-02 값으로 채움
                        .groupby('country')['confidence_index'].first())
        set_df['confidence_index'] = set_df['confidence_index'].fillna(set_df['country'].map(feb_vals))
        return set_df.drop(columns='month')

    '''
    ========================================
    3. STL 분해 및 이벤트 구간 탐지
    ========================================
    @ stl_decompose: 도시별 수요를 STL 분해하여 trend, seasonal, residual를 계산

    @ detect_event_periods: STL 분해 결과 잔차가 mean + 1.5*std 이상으로 14일 이상 연속되는 구간을 탐색
    -------------------------------------------

    '''
    def stl_decompose(city_demand_df):
        outs = []
        for city in city_demand_df['city'].unique():
            g = city_demand_df[city_demand_df['city']==city].set_index('date')['demand_sum'] # 도시의 수요 시계열 추출
            g.index = pd.to_datetime(g.index)
            g = g.asfreq('D') # 일 단위로 정규화
            country = city_demand_df[city_demand_df['city']==city]['country'].values[0] # [0]은 국가 약어
            res = STL(g, period=365, robust=True).fit() # STL 분해
            outs.append(pd.DataFrame({
                'date': g.index, 'city': city, 'country': country,
                'demand_sum': res.observed, 'trend': res.trend,
                'seasonal': res.seasonal, 'residual': res.resid
            }))
        return pd.concat(outs, ignore_index=True)

    def detect_event_periods(stl_df):
        ev = []
        for country, grp in stl_df.groupby('country'):
            g = grp.sort_values('date').copy()
            g['abs_residual'] = g['residual'].abs() # 잔차의 절댓값 계산
            thr = g['abs_residual'].mean() + 1.5*g['abs_residual'].std() # 임계치로 mean + 1.5*std
            g['is_spike'] = g['abs_residual'] > thr
            g['spike_group'] = (g['is_spike'] != g['is_spike'].shift()).cumsum() # 연속으로 spike 하는 구간 그룹 식별
            for _, s in g[g['is_spike']].groupby('spike_group'):
                start, end = s['date'].min(), s['date'].max() # 그룹의 최대,최소, 차이 계산
                dur = (end - start).days + 1
                if dur >= 14: # 지속기간 2주 이상
                ev.append({'country':country,'event_start':start,'event_end':end,'duration_days':dur})
        return pd.DataFrame(ev)


    '''
    -------------- 특성/이벤트 플래깅 --------------
    
    @ make_X: 비연속형 변수들(카테고리 변수)를 원핫 인코딩

    @ build_event_candidates_2324
        - 입력
            set_df: dataframe, 데이터 모두 merge한 통합 데이터셋
            event_summary_df: dataframe, stl 잔차 기반으로 산출한 이벤트 구간
        - 설명
            0) 데이터 수치화하여 요인들 생성, 2324년도 필터링
            1) mean + 2*std로 threshold 설정
            2) flag_block: 날짜별로 threshold 넘는 요인이 25% 이상이면 이벤트로 판단(is_event=1)
            3) extract_periods: 이벤트 중 14일 이상 연속되는 구간을 탐색
        - 출력
            event_candidates_2324: dataframe, 2324에서 요인들의 급증을 기준으로 이벤트 구간을 계산
    -------------------------------------------
    '''
    def make_X(df, id_cols, target=None):
        num = df.select_dtypes(include=[np.number]).copy() # 숫자형 변수
        cat = df.select_dtypes(include=['object','category']).copy() # 범주형 변수
        drop_cols = [c for c in id_cols + ([target] if target else []) if c in num.columns] # id랑 target은 제외하고 변환
        num = num.drop(columns=drop_cols, errors='ignore')
        cat = cat.drop(columns=[c for c in id_cols if c in cat.columns], errors='ignore')
        dmy = pd.get_dummies(cat, drop_first=True, dtype=int) # 범주형 원핫 인코딩
        return pd.concat([num, dmy], axis=1)

    def build_event_candidates_2324(set_df, event_summary_df):
        ID_COLS = ['date','country']

        # 0) 요인 테이블
        X_all = make_X(set_df, id_cols=ID_COLS)
        base  = set_df[['date','country']].reset_index(drop=True)
        Xfull = pd.concat([base, X_all.reset_index(drop=True)], axis=1)
        feat_cols = [c for c in Xfull.columns if c not in ID_COLS]

        df_2324 = Xfull[Xfull['date'].dt.year.isin([2023, 2024])].copy()

        # 1) 글로벌/국가별 threshold
        k = 2.0
        _ev_parts = []
        for _, r in event_summary_df.iterrows():
            m = (Xfull['country'].eq(r['country']) & Xfull['date'].between(r['event_start'], r['event_end']))
            _ev_parts.append(Xfull.loc[m, feat_cols])
        global_ev  = pd.concat(_ev_parts, axis=0) if _ev_parts else Xfull.iloc[0:0, :]
        global_thr = global_ev.mean().add(k * global_ev.std(ddof=0))

        thr_by_country = {}
        for cty, evs in event_summary_df.groupby('country'):
            ev_parts = []
            for _, r in evs.iterrows():
                m = (Xfull['country'].eq(cty) & Xfull['date'].between(r['event_start'], r['event_end']))
                ev_parts.append(Xfull.loc[m, feat_cols])
            ev_df = pd.concat(ev_parts, axis=0) if ev_parts else None
            thr_by_country[cty] = (global_thr if (ev_df is None or ev_df.empty)
                                   else ev_df.mean().add(k * ev_df.std(ddof=0)))



        # 2) 국가별 플래깅 (그룹명 cty를 명시적으로 전달)
        def flag_block(g, cty):
            thr = thr_by_country.get(cty, global_thr).reindex(feat_cols)
            thr = thr.fillna(np.inf)
            over = g[feat_cols].gt(thr, axis=1).mean(axis=1) # threshold 넘는 비율 계산
            g = g.copy()
            g['is_event'] = (over >= 0.25).astype(int)  # 25% 이상 넘으면 이벤트
            return g

        parts = []
        for cty, g in df_2324.groupby('country'):
            g2 = flag_block(g.copy(), cty)
            g2['country'] = cty  # 그룹 키 유지
            parts.append(g2)
        df_2324 = pd.concat(parts, ignore_index=True) if parts else df_2324.assign(is_event=0)

        daily = (df_2324.groupby(['country','date'], as_index=False).agg(is_event=('is_event', 'max'))) # 하루에 하나의 is_event만 남김

        # 3) 연속 구간 추출
        inclusive = False
        min_run   = 14

        def extract_periods(g):
            g = g.sort_values('date').reset_index(drop=True)
            g['grp'] = (g['is_event'].ne(g['is_event'].shift())).cumsum()
            outs = []
            for _, b in g.groupby('grp'):
                if b['is_event'].iat[0] != 1:
                    continue
                start = b['date'].min()
                end   = b['date'].max()
                days  = (end - start).days + (1 if inclusive else 0)
                if days >= min_run:
                    outs.append({
                        'country': b['country'].iat[0],'event_start': start,'event_end': end, 'duration_days': days})
            return pd.DataFrame(outs)

        outs = []
        for cty, g in daily.groupby('country'):
            g = g.copy()
            g['country'] = cty
            ext = extract_periods(g)
            if not ext.empty:
                outs.append(ext)

        if outs:
            return pd.concat(outs, ignore_index=True)
        else:
            return pd.DataFrame(columns=['country','event_start','event_end','duration_days'])

    '''
    ========================================
    4. 변수 선택
    ========================================
    @ vif_filter: VIF가 10 이하인 것만 남도록 순차적으로 필터링
    -------------------------------------------
    '''
    def vif_filter(df, th=10.0):
        cols = df.columns.tolist()
        if len(cols) <= 1:
            return cols
        while True:
            Z = sm.add_constant(df[cols], has_constant='add') # 상수항 추가
            vifs = pd.Series([variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])], index=cols) # VIF 계산
            if vifs.max() <= th: # 순차적으로 제거해서 VIF 10이하만 남기기
                break
            cols.remove(vifs.idxmax())
            if len(cols) <= 1: 
                break
        return cols

    '''
    ========================================
    5. 수요 예측
    ========================================
    @ event_mask: Boolean Series, date에서 events_df에 따라 이벤트 기간인지 여부를 True/False로 표시 

    @ build_non_event_stats: DataFrame, 이벤트 구간이 아닌 날의 통계를 계산, tbl에 mu와 sigma

    @ fetch_mu_sigma: tuple, 특정 국가의 mu와 sigma 가져오기 
    -------------------------------------------
    '''
    def event_mask(dts, country, events_df):
        m = pd.Series(False, index=dts.index)
        evs = events_df[events_df['country'].eq(country)]
        for _, r in evs.iterrows():
            m |= dts.between(r['event_start'], r['event_end']) # 구간 안에 있으면 true
        return m

    def build_non_event_stats(df, cutoff, events_df):
        hist = df[df['date'] <= cutoff].copy()
        hist['is_event'] = False
        for c in hist['country'].unique():
            idx = hist.index[hist['country'].eq(c)]
            m = event_mask(hist.loc[idx,'date'], c, events_df) # 날짜별 이벤트 여부
            hist.loc[m.index, 'is_event'] = m.values
        ne = hist[~hist['is_event']] # 비이벤트 구간만 추출
        tbl = (ne.groupby('country')['demand']
                 .agg(mu='mean', sigma=lambda x: x.std(ddof=0)).reset_index()) # 비이벤트에서 나라별 평균과 표준편차 계산
        return tbl

    def fetch_mu_sigma(country, tbl_country):
        m = tbl_country['country'].eq(country)
        if m.any():
            r = tbl_country[m].iloc[0]
            return float(r['mu']), float(r['sigma'])
        return None, None

    '''
    ---------------- 예측 ----------------------
    @ forecast
        - 입력
            df: dataframe, 18-24까지의 모든 feature 포함, 23-24의 demand는 na로 처리
            events_df: dataframe, 수요 급증을 바탕으로 한 18-22의 이벤트 구간 + 요인을 바탕으로 예측한 2324의 이벤트 구간
        - 설명
            1) 그룹 분리, lag 생성: 도시 * sku별로 그룹 분리, 365일 lag 생성
            2) 회귀모델 학습: lag demand와 feature들 입력으로 하는 Ridge 회귀 사용 (alpha는 GridSearch로 최적 선택)
            3) 이벤트 구간 보정 목표 설정: 비이벤트 평균 + 2σ 수준을 이벤트 보정의 목표 레벨로 사용
            4) 예측: 학습한 모델로 예측, 이벤트 구간일 경우 예측치에 보정치 곱하여 보정
        - 출력
            out_df: dataframe, 2324년도의 도시별 날짜별 sku별 demand 예측치
    -------------------------------------------
    '''
    def forecast(df, events_df):
        need_cols = {'date','country','city','sku','demand','month','dow'}
        assert need_cols.issubset(df.columns)
        num_cols  = df.select_dtypes(include=['number']).columns.tolist()
        exog_cols = [c for c in num_cols if c != 'demand']
        cutoff = df.loc[df['demand'].notna(), 'date'].max()
        gcols  = ['city','sku']

        tbl_country = build_non_event_stats(df, cutoff, events_df) # 비이벤트 구간의 평균과 표준편차
        outs = []
        
        # 1) 그룹 분리, lag 생성
        for (g_city, g_sku), g in df.sort_values('date').groupby(gcols):
            g = g.copy()
            if exog_cols:
                g[exog_cols] = g[exog_cols].ffill().bfill() # feature 결측치 보간
            g['lag_365'] = g['demand'].shift(365) # 일년 전 동일일 수요

            
            # 2) 회귀 모델 학습
            tr = g[g['date'] <= cutoff].dropna(subset=['demand','lag_365'])
            Xtr = tr[['lag_365'] + exog_cols]
            ytr = tr['demand'].astype(float)
            
            # Ridge 회귀 + alpha 최적화 (GridSearchCV)
            gs = GridSearchCV(Ridge(fit_intercept=True),
                              param_grid={'alpha':[0.1,0.5,1.0,2.0,5.0,10.0]},
                              scoring='neg_mean_squared_error', cv=3, n_jobs=-1).fit(Xtr, ytr)
            model = gs.best_estimator_

            
            # 3) 이벤트 구간 배수 보정 목표 설정
            country = g.iloc[0]['country']
            mu, sigma = fetch_mu_sigma(country, tbl_country)
            target_lvl = None if (mu is None or sigma is None) else (mu + 2.0 * sigma) # μ + 2σ 이상으로 보정

            g['pred'] = np.nan

            # 4) 2023 예측
            m23 = (g['date'] > cutoff) & (g['date'].dt.year == 2023)
            if m23.any():
                X23 = g.loc[m23, ['lag_365'] + exog_cols]
                y23 = model.predict(X23)
                g.loc[m23, 'pred'] = np.clip(y23, 0, None)
                if target_lvl is not None:
                    ev23 = event_mask(g.loc[m23,'date'], country, events_df)
                    # 이벤트 구간이면 uplift 적용
                    if ev23.any():
                        med = np.median(g.loc[m23].loc[ev23.values,'pred'])
                        if med and med > 0:
                            uplift = max(1.0, target_lvl / med)  # 목표 수준까지 배수 조정
                            idx = g.loc[m23].index[ev23.values]
                            g.loc[idx,'pred'] *= float(uplift)
                g.loc[m23,'demand'] = g.loc[m23,'pred']  # 예측치 demnand에 추가하여 2024 lag 반영

            # 4) 2024 예측
            g['lag_365'] = g['demand'].shift(365)
            m24 = (g['date'] > cutoff) & (g['date'].dt.year == 2024)
            if m24.any():
                X24 = g.loc[m24, ['lag_365'] + exog_cols].copy()
                X24['lag_365'] = (X24['lag_365']
                                  .fillna(method='ffill')
                                  .fillna(method='bfill')
                                  .fillna(Xtr['lag_365'].mean()))
                y24 = model.predict(X24)
                g.loc[m24, 'pred'] = np.clip(y24, 0, None)

                if target_lvl is not None:
                    ev24 = event_mask(g.loc[m24,'date'], country, events_df)
                    if ev24.any():
                        med = np.median(g.loc[m24].loc[ev24.values,'pred'])
                        if med and med > 0:
                            uplift = max(1.0, target_lvl / med)
                            idx = g.loc[m24].index[ev24.values]
                            g.loc[idx,'pred'] *= float(uplift)

            fu = g[g['date'] > cutoff][['date','sku','city','pred']].rename(columns={'pred':'demand'})
            outs.append(fu)

        out_df = (pd.concat(outs, ignore_index=True)
                  .sort_values(['city','sku','date']).reset_index(drop=True))
        return out_df















    
    '''======================================== 파이프라인 ========================================'''
    '''
    -------------------------------------------
    1. module import & load raw data
    -------------------------------------------
    '''
    dfs = load_csvs_to_dict(args.data_path)

    ######################## 외부 데이터 로드
    try:
        site_candidates_df     = dfs['site_candidates_df']
        oil_price_df           = dfs['oil_price_df']
        consumer_confidence_df = dfs['consumer_confidence_df']
        currency_df            = dfs['currency_df']
        weather_df             = dfs['weather_df']
        marketing_spend_df     = dfs['marketing_spend_df']
        holiday_lookup_df      = dfs['holiday_lookup_df']
        calendar_df            = dfs['calendar_df']
        labour_policy_df       = dfs['labour_policy_df']
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(f"[data_path]에 '{missing}.csv'가 존재하는지 확인하세요.")

    ######################## 수요 로드
    demand_df, city_country_map, city_demand_df = load_demand(args.sqlite_path, site_candidates_df)
    
    '''
    -------------------------------------------
    2. 전처리 후 외부 데이터 merge 
    -------------------------------------------
    '''
    ######################## 결측치 대체
    oil_price_df = prepare_oil(oil_price_df)
    consumer_confidence_df = prepare_consumer_confidence(consumer_confidence_df)
    currency_df = prepare_currency(currency_df, oil_price_df)

    ######################## 외부 데이터 merge: set_df(날짜 * 도시)
    dates = pd.date_range('2018-01-01','2024-12-31',freq='D')
    set_df = pd.DataFrame({'date': dates}).merge(city_country_map, how='cross') # set_df의 구조로 날짜, 도시, 나라 통일
    set_df = merge_base_frames(set_df, weather_df, marketing_spend_df, calendar_df, oil_price_df)
    set_df = merge_holiday(set_df, holiday_lookup_df)
    set_df = merge_currency(set_df, currency_df)
    set_df = merge_labour(set_df, labour_policy_df)
    set_df = merge_cc_with_lag(set_df, consumer_confidence_df)
    set_df = set_df.sort_values(['date','country','city']).reset_index(drop=True)
    
    '''
    -------------------------------------------
    3. STL 분해 및 이벤트 구간 탐지
    -------------------------------------------
    '''
    ######################## 18-22: STL 분해, 잔차로 이벤트 구간 탐지
    stl_results = stl_decompose(city_demand_df)
    event_summary_df = detect_event_periods(stl_results)

    ######################## 23-24: 특성 기반 이벤트 구간 탐지
    event_candidates_2324 = build_event_candidates_2324(set_df, event_summary_df)

    ######################## 18-24: 이벤트 전체 구간 merge
    events_df = pd.merge(event_summary_df, event_candidates_2324,
                         on=['country','event_start','event_end','duration_days'],how='outer').sort_values('event_start')
    
    '''
    -------------------------------------------
    4. 변수 선택
    -------------------------------------------
    '''

    ######################## 18-22: 요인과 수요 데이터의 관계 파악
    ######################## 상관관계 0.1 이상, VIF 10 이하로 변수 선택(selected_cols)
    END_DATE = '2022-12-31'; ID_COLS = ['date','city','country']; TARGET = 'demand_sum'
    
    left  = set_df.loc[set_df['date'] <= '2022-12-31'].copy()
    right = city_demand_df.loc[city_demand_df['date'] <= END_DATE, ['date','city',TARGET]].copy()
    df_sel = left.merge(right, on=['date','city'], how='inner')

    X_sel = make_X(df_sel, id_cols=['date','city','country'], target=TARGET) # 수치형으로 변환
    X_sel = X_sel.loc[:, X_sel.notna().all()] # 결측치 제외
    X_sel = X_sel.loc[:, X_sel.nunique(dropna=False) > 1] # 상수 변수 제외
    y = df_sel[TARGET]

    corr = X_sel.apply(lambda s: s.corr(y))
    keep_corr = corr.index[corr.abs() >= 0.10].tolist() # 상관관계 0.1 이상
    X_sel = X_sel[keep_corr]
    keep_vif = vif_filter(X_sel, 10.0) if X_sel.shape[1] > 1 else X_sel.columns.tolist() # VIF 10 이하
    selected_cols = keep_vif

    ######################## 예측용 df 생성
    ######################## 18-24까지의 모든 sku* 날짜별 feature 포함, 23-24의 demand는 na로 처리, 시계열 feature 추가
    X_all = make_X(set_df, id_cols=ID_COLS, target=None)
    exog_df = pd.concat([set_df[ID_COLS], X_all[[c for c in selected_cols if c in X_all.columns]]], axis=1) # 날짜 * 도시별 요인 추가
    sku_map = demand_df[['city','sku','country']].drop_duplicates()  # sku 파악
    grid = exog_df.merge(sku_map, on=['city','country'], how='left') # sku로 확장
    df = grid.merge(demand_df, on=['date','city','sku','country'], how='left') # sku 확장한 요인과 수요 left merge(2324의 demand는 na)

    df['time_idx'] = (df['date'] - df['date'].min()).dt.days.astype('int32') # 시계열 feature로 time_idx, month, dow 추가
    df['month'] = df['date'].dt.month.astype('int16')
    df['dow']   = df['date'].dt.dayofweek.astype('int16')
    for c in ['country','city','sku']:
        df[c] = df[c].astype('category')

    '''
    -------------------------------------------
    5. 예측
    -------------------------------------------
    '''
    out_df = forecast(df.copy(), events_df)
    ######################## 제출 형식에 따른 조정
    out_df = out_df.rename(columns={'demand':'mean'}) # 컬럼 이름 mean
    out_df['mean'] = out_df['mean'].round().astype(int) # 정수로 round
    out_df.to_csv(args.out_csv, index=False)
    print(f"[ok] saved -> {args.out_csv}  rows={len(out_df)}")


if __name__ == "__main__":
    main()
