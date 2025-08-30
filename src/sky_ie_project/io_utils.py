import pandas as pd
import sqlite3
import math
from collections import defaultdict
from datetime import datetime, timedelta

def get_fc_info(fc_sol):
    """
    fc_sol: {"M": {city: 1, ...}, "X": {(f,w): 1, ...}, "P": {(f,w,u):n, ...}}
    return
      fc_loc: set of factory locations
      fc_wh_edge: set of (factory(from), warehouse(to)) edges
    """
    M_sol = fc_sol.get("M", {})
    X_sol = fc_sol.get("X", {})
    P = fc_sol.get("P", {})
    fc_loc = {f for f, v in M_sol.items() if v}  # 값(1) 버리고 노드 정보만 추출
    fc_wh_edge = {e for e, v in X_sol.items() if v}  # 값(1) 버리고 엣지 정보만 추출
    
    return fc_loc, fc_wh_edge, P

def get_wh_info(wh_sol):
    """
    wh_sol: {"N": {city: 1, ...}, "Y": {(w,k): 1, ...}}
    return
      wh_loc: set of warehouse locations
      wh_ct_edge: set of (warehouse(from), city(to)) edges
    """
    N_sol = wh_sol.get("N", {})
    Y_sol = wh_sol.get("Y", {})
    wh_loc = {w for w, v in N_sol.items() if v}  # 값(1) 버리고 노드 정보만 추출
    wh_ct_edge = {e for e, v in Y_sol.items() if v}  # 값(1) 버리고 엣지 정보만 추출
    
    return wh_loc, wh_ct_edge

def aggregate_demand_by_wh(wh_loc, wh_ct_edge, daily_demand, skus):
    """
    demand_wh_u[(w,u)] = sum_{k: (w,k) in wh_ct_edge} daily_demand[(k,u)]
    - daily_demand: (city, sku) -> int (일간 평균, 올림 처리된 값)
    - skus: sku 리스트
    """
    demand_wh_u = defaultdict(int)
    # 창고가 없거나 연결 도시가 없으면 0
    for (w, k) in wh_ct_edge:
        for u in skus:
            demand_wh_u[(w, u)] += int(daily_demand.get((k, u), 0))
    
    for w in wh_loc:
        for u in skus:
            _ = demand_wh_u[(w, u)]
    return dict(demand_wh_u)

# information about each site including warehouse and factory
# Output: wh_info(dataframe), fc_info(dataframe)
# wh_info: warehouse information
# fc_info: factory information including carbon factor
def get_site_info(
    candidates_path = "data/site_candidates.csv", 
    init_cost_path = "data/site_init_cost.csv", 
    carbon_factor_path = "data/carbon_factor_prod.csv"
):
    # load data
    candidates = pd.read_csv(candidates_path)
    init_costs = pd.read_csv(init_cost_path, usecols = ["site_id", "init_cost_usd"])
    carbon_factors = pd.read_csv(carbon_factor_path)
    carbon_factors = carbon_factors.rename(columns = {"factory": "site_id"}) # rename for merge compatibility
    
    # Warehouse/Factory 정보 데이터프레임 생성
    merged = candidates.merge(init_costs, on = "site_id", how = "inner")
    wh_info = merged[merged["site_type"] == "warehouse"].reset_index(drop = True)
    fc_info = merged.merge(carbon_factors, on = "site_id", how = "inner").reset_index(drop = True)
    
    return wh_info, fc_info

# Output: sku_info(dataframe)
def get_sku_info(
    spec_path = "data/sku_meta.csv",
    labour_path = "data/labour_requirement.csv",
    inv_cost_path = "data/inv_cost.csv",
    short_cost_path = "data/short_cost.csv"
):    
    # load data
    spec = pd.read_csv(spec_path)
    labour = pd.read_csv(labour_path)
    inv_cost = pd.read_csv(inv_cost_path)
    short_cost = pd.read_csv(short_cost_path)
    
    # sku 정보 데이터프레임 생성
    sku_info = spec.merge(labour, on = "sku", how = "inner") \
                   .merge(inv_cost, on = "sku", how = "inner") \
                   .merge(short_cost, on = "sku", how = "inner")
    
    return sku_info

# average daily demand table for each city and sku
# Output: avg_daily_demand(dict)
def get_avg_demand(
    fc_info,
    db_path = "data/demand_train.db",
    launch_path = "data/sku_meta.csv"
):
    """
    각 도시 및 SKU에 대해 '일간 수요의 산술평균'을 계산하여 반환.
    - 주간 집계/연도 가중치 사용하지 않음
    - 출시일 이전 데이터는 제외
    - 결과는 (city, sku) -> int (양수는 올림, 그 외 0)
    """
    # 1) DB 로드
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT date, city, sku, demand FROM demand_train", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])

    # 2) SKU 출시일 병합 및 필터링
    launch_date_df = pd.read_csv(launch_path)
    launch_date_df["launch_date"] = pd.to_datetime(launch_date_df["launch_date"])
    df = df.merge(launch_date_df[["sku", "launch_date"]], on = "sku", how = "left")
    df = df[df["date"] >= df["launch_date"]]

    # 3) 도시×SKU 일간 수요의 '산술평균' 계산
    daily_mean = (
        df.groupby(["city", "sku"], as_index = False)["demand"]
          .mean()
          .rename(columns={"demand": "daily_mean"})
    )

    # 4) 피벗 → 도시 순서 맞추기 → 결측 0
    city_list = fc_info["city"].tolist()
    skus = launch_date_df["sku"].unique().tolist()

    mat = (
        daily_mean.pivot(index = "city", columns = "sku", values = "daily_mean")
                 .reindex(index = city_list, columns = skus)
                 .fillna(0)
    )

    # 5) 양수는 올림, 그 외 0 → dict[(city, sku)] = int
    mat = mat.applymap(lambda x: math.ceil(x) if x > 0 else 0)
    return mat.stack().to_dict()

def get_capacity(date, path = "data/factory_capacity.csv"): 
    df = pd.read_csv(path, parse_dates = ["week"])
    date = pd.to_datetime(date)
    df = df[df["week"] == date]
    return df  # 입력받은 date가 속한 week의 모든 공장 capacity 반환

# Output: mt_cost(dict)
def get_mt_cost(
    fc_info, 
    mt_cost_path = "data/prod_cost_excl_labour.csv"
):
    mt_cost = pd.read_csv(mt_cost_path)
    mt_cost = mt_cost.rename(columns = {"factory": "site_id"}) # rename for merge compatibility
    site_info = fc_info[["site_id", "city"]].copy() # 필요한 정보만 추출
    
    mt_cost = mt_cost.merge(site_info, on = "site_id", how = "inner")
    mt_cost = mt_cost.pivot(index = "city", columns = "sku", values = "base_cost_usd")
    city_list = fc_info["city"].tolist()
    mt_cost = mt_cost.reindex(city_list)
    mt_cost = mt_cost.stack().to_dict()
    
    return mt_cost

# Output: wage(dict)
def get_wage(
    fc_info,
    labour_path = "data/labour_policy.csv",
    currency_path = "data/currency.csv",
    year_weights = {2018: 0.2, 2019: 0.2, 2020: 0.2, 2021: 0.2, 2022: 0.2}
):
    # 1. 시급 데이터 로드
    labour = pd.read_csv(labour_path)
    labour["currency"] = labour["currency"].apply(lambda x: f"{x}=X" if not x.endswith("=X") else x)
    labour = labour[labour["year"] <= 2022]

    # 2. 환율 데이터 로드
    fx = pd.read_csv(currency_path)
    fx["Date"] = pd.to_datetime(fx["Date"])
    fx["year"] = fx["Date"].dt.year
    fx = fx[fx["year"] <= 2022]

    # 3. 연도별 환율 평균 계산
    fx_avg = fx.drop(columns=["Date"]).groupby(fx["year"]).mean().T  # index: currency, columns: year

    # 4. 연도별 가중 평균 환율 계산
    norm_weights = {y: w / sum(year_weights.values()) for y, w in year_weights.items() if y in fx_avg.columns}
    fx_avg["weighted"] = fx_avg[list(norm_weights.keys())] @ pd.Series(norm_weights)

    # 5. 국가별 환율 merge
    labour = labour.merge(fx_avg["weighted"], left_on = "currency", right_index = True, how = "left")
    labour.loc[labour["country"] == "USA", "weighted"] = 1.0

    # 6. 환율 적용: local wage ÷ (현지통화 per USD) → USD wage
    labour["usd_wage"] = labour["regular_wage_local"] / labour["weighted"]

    # 7. 국가별 평균 USD wage 산출
    wage = (
        labour.groupby("country")["usd_wage"]
        .mean()
        .reset_index(name = "wage_usd")
    )

    city_country_map = dict(zip(fc_info["city"], fc_info["country"]))
    wage = dict(zip(wage["country"], wage["wage_usd"]))
    wage = {city: wage[city_country_map[city]] for city in fc_info["city"]}
    
    return wage  # city, wage_usd

# Input: fc_info
# Output: tp_cost(dict), tp_carbon(dict), tp_leadtime(dict)
# tp_cost_(i, j) = 12 * delta_(i, j) * psi_(i, j) + beta_(i, j) : 운송 비용
# tp_carbon_(i, j) = 0.4 * delta_(i, j) * psi'_(i, j) : 운송 탄소 배출량
def get_tp_info(fc_info):
    city_info = fc_info[["city", "country", "lat", "lon"]].copy() # 필요한 정보만 추출
    city_list = city_info["city"].tolist()
    tp_factor = pd.read_csv("data/transport_mode_meta.csv")
    
    tp_cost = pd.DataFrame(
        data = 12.0,
        index = city_list,
        columns = city_list
    )
    tp_carbon = pd.DataFrame(
        data = 0.4,
        index = city_list,
        columns = city_list
    )
    tp_mode = pd.DataFrame(
        data = "TRUCK",
        index = city_list,
        columns = city_list        
    )
    tp_leadtime = pd.DataFrame(
        data = 0,
        index = city_list,
        columns = city_list
    )
    # 하버사인 방식 기반 두 지점 간 거리 계산 함수
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371 # 지구 반지름

        # 위도, 경도 값 라디안으로 변환
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance
    
    # tp_cost & tp_carbon & tp_leadtime 값 채우기
    # i < j인 경우에만 계산하고 대칭적으로 값 채움 (∵ symmetric matrix → (i, j) == (j, i))
    for i in range(len(city_list)):
        for j in range(i + 1,len(city_list)):
            lat1, lon1 = city_info.iloc[i]["lat"], city_info.iloc[i]["lon"]
            lat2, lon2 = city_info.iloc[j]["lat"], city_info.iloc[j]["lon"]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            
            # 국내/국제/EU 여부와 거리에 따른 tp_mode 결정
            # 국내 - TRUCK
            if city_info.iloc[i]["country"] == city_info.iloc[j]["country"]: 
                tp_mode.at[city_list[i], city_list[j]] = "TRUCK"
                
            # EU - 1000km 이하: SHIP, 1000km 초과: TRUCK
            elif {city_info.iloc[i]["country"], city_info.iloc[j]["country"]} == {"DEU", "FRA"}:
                if distance <= 1000:
                    tp_mode.at[city_list[i], city_list[j]] = "TRUCK"
                else:
                    tp_mode.at[city_list[i], city_list[j]] = "SHIP"
            
            # 국제(Non-EU) - 1000km 이하: SHIP, 초과: AIR
            else:
                if distance <= 1000:
                    tp_mode.at[city_list[i], city_list[j]] = "SHIP"
                else:
                    tp_mode.at[city_list[i], city_list[j]] = "AIR"
            
            match distance:
                case d if d <= 500:
                    tp_leadtime.at[city_list[i], city_list[j]] = 2
                case d if d <= 1000:
                    tp_leadtime.at[city_list[i], city_list[j]] = 3
                case d if d <= 2000:
                    tp_leadtime.at[city_list[i], city_list[j]] = 5
                case _:
                    tp_leadtime.at[city_list[i], city_list[j]] = 8
            
            factor = tp_factor.loc[tp_factor["mode"] == tp_mode.at[city_list[i], city_list[j]]]
            
            tp_cost.at[city_list[i], city_list[j]] = 12 * float(factor["cost_per_km_factor"].iloc[0]) * distance
            + 4000 * (not (city_info.iloc[i]["country"] == city_info.iloc[j]["country"] or 
                    {city_info.iloc[i]["country"], city_info.iloc[j]["country"]} == {"DEU", "FRA"}))
            tp_carbon.at[city_list[i], city_list[j]] = 0.4 * float(factor["co2_per_km_factor"].iloc[0]) * distance
            tp_leadtime.at[city_list[i], city_list[j]] = math.ceil(tp_leadtime.at[city_list[i], city_list[j]] * factor["leadtime_factor"].iloc[0])
            
            tp_cost.at[city_list[j], city_list[i]] = tp_cost.at[city_list[i], city_list[j]]
            tp_carbon.at[city_list[j], city_list[i]] = tp_carbon.at[city_list[i], city_list[j]]
            tp_leadtime.at[city_list[j], city_list[i]] = tp_leadtime.at[city_list[i], city_list[j]]
            tp_mode.at[city_list[j], city_list[i]] = tp_mode.at[city_list[i], city_list[j]]
    
    tp_cost = tp_cost.stack().to_dict()
    tp_carbon = tp_carbon.stack().to_dict()
    tp_leadtime = tp_leadtime.stack().to_dict()
    tp_mode = tp_mode.stack().to_dict()
    
    return tp_cost, tp_carbon, tp_leadtime, tp_mode

# 미완성 (화폐 단위 통일)
"""def local_wage_to_usd(date, unit_price, currency_path = "data/currency.csv"):
    # Load currency data
    currency_data = pd.read_csv(currency_path)
    currency_data["Date"] = pd.to_datetime(currency_data["Date"])
    
    # Filter for the specific date and city
    currency_data = currency_data[currency_data["Date"] == date]
    if city not in currency_data.columns:
        raise ValueError(f"Currency data for {city} not found.")
    
    # Get the exchange rate for the city
    exchange_rate = currency_data[city].values[0]
    
    # Convert local price to USD
    return unit_price / exchange_rate"""

# currency.csv는 date가 아니고 Date여서 차후 점검 필요
def fill_na(path):
    df = pd.read_csv(path, parse_dates = ["date"])
    df = df.sort_values("date").set_index("date")
    full_idx = pd.date_range(start = "2018-01-01", end = "2024-12-31", freq = "D")
    df = df.reindex(full_idx).ffill()
    df = df.reset_index().rename(columns = {"index": "date"})
    
    return df

def city_to_country(city) -> dict[str, str]:
    _, fc_info = get_site_info()
    map = dict(zip(fc_info["city"], fc_info["country"]))
    
    return map.get(city)

def city_to_id(city, type) -> dict[str, str]:
    wh_info, fc_info = get_site_info()
    if type == "factory":
        fc_id = dict(zip(fc_info["city"], fc_info["site_id"]))
        return fc_id.get(city)
    else:
        wh_id = dict(zip(wh_info["city"], wh_info["site_id"]))
        return wh_id.get(city)

def id_to_city(id) -> dict[str, str]:
    _, fc_info = get_site_info()
    map = dict(zip(fc_info["site_id"], fc_info["city"]))
    
    return map.get(id)

def week_start(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    monday = dt - pd.Timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")