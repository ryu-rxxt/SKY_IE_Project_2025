from domain import *
from io_utils import *
from flag import *

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

DEMAND_STORE = None

# (date: str, city: str): 각 sku 수요를 원소로 가지는 벡터(len == 25)를 메모리에 적재
# 시간 복잡도 및 불필요한 IO 접근을 줄여 병목 현상 방지
class DemandStore:
    def __init__(self, sku_to_idx: dict[str, int]):
        self.sku_to_idx = sku_to_idx
        self.map: dict[tuple[str, str], np.ndarray] = {}
        
    def load(self,
             db_path = "data/demand_train.db",
             csv_path = "data/forecast_submission_template.csv",
             start = "2018-01-01",
             end = "2024-12-31"
    ):
        # 1) 2018-01-01 ~ 2022-12-31 관측치
        with sqlite3.connect(db_path) as conn:
            hist = pd.read_sql_query(
                """
                SELECT date, city, sku, demand
                FROM demand_train
                WHERE date BETWEEN ? AND ?
                """,
                conn,
                params = (start, "2022-12-31"),
            )
        
        # 2) 2023-01-01 ~ 2024-12-31 예측치
        fc = pd.read_csv(csv_path).rename(columns = {"mean": "demand"})
        fc = fc[(fc["date"] >= "2023-01-01") & (fc["date"] <= end)]
        fc = fc[["date", "city", "sku", "demand"]]
        
        df = pd.concat([hist, fc], ignore_index = True)
        
        # (date, city) 벡터
        for (date, city), group in df.groupby(["date", "city"], sort = False):
            vector = np.zeros(25, dtype = int)
            
            for row in group.itertuples(index = False):
                vector[self.sku_to_idx[row.sku]] = int(row.demand)
            
            self.map[(date, city)] = vector
    
    def get(self, date: str, city: str) -> np.ndarray:
        vector = self.map.get((date, city))
        
        if vector is None:
            return np.zeros(25, dtype = int)
        
        if date < "2023-01-01":
            return np.ceil(0.95 * vector).astype(int)
        
        return vector.copy()
        
# P: {(i, w, u): qty, ...} -> qty > 0이면 '해당 공장 i가 창고 w로 해당 SKU u를 보낸다'로 간주
# return: dict[w][i] = np.ndarray(shape = (25,), dtype = int)  # 이진 벡터(보내면 1, 아니면 0)
def get_supplier_vector(
    P: dict, 
    sku_list: list[str], 
    sku_to_idx: dict[str, int]
):
    supplier_vector = defaultdict(lambda: defaultdict(lambda: np.zeros(len(sku_list), dtype = int)))
    
    for (i, w, u), qty in P.items():
        if qty and qty > 0:
            idx = sku_to_idx[u]
            supplier_vector[w][i][idx] = 1
    
    return supplier_vector

# Generate Factory/Warehouse/City Object Instance & initial setting
def init_setting(wh_sol, fc_sol, city_list, sku_list, sku_to_idx):
    # 공장/창고 최적해로부터 필요 정보 가공
    wh_loc, wh_ct_edge = get_wh_info(wh_sol)
    fc_loc, fc_wh_edge, P = get_fc_info(fc_sol)
    
    wh_clients = defaultdict(list)
    for (w, k) in wh_ct_edge:
        wh_clients[w].append(k)
    sup_info = {k: w for (w, k) in wh_ct_edge}

    # P를 '0-1 벡터'로 변환
    supplier_vector = get_supplier_vector(P, sku_list, sku_to_idx)
    
    # Factory 인스턴스 생성
    factories = {}
    for f in fc_loc:
        # wh_links는 필요시 유지/생략 가능. 여기선 비워둠.
        factories[f] = Factory(
            city = f,
            open_date = "2018-01-08"
        )

    # Warehouse 인스턴스 (supplier를 '이진 벡터'로 세팅)
    warehouses = {}
    for w in wh_loc:
        # supplier_vector[w]가 없으면 빈 dict
        sup_dict = {i: vec for i, vec in supplier_vector.get(w, {}).items()}
        warehouses[w] = Warehouse(
            city = w,
            open_date = "2018-01-01",
            client = sorted(wh_clients.get(w, [])),
            supplier = sup_dict,                          # dict[str, np.ndarray(0/1)]
            stock = np.full(25, 2000, dtype = int),
            dlv_in_prog = {}
        )

    # City 인스턴스
    cities = {}
    for c in city_list:
        sup = sup_info.get(c, "")
        cities[c] = City(
            city = c,
            dlv_in_prog = {}
        )

    return factories, warehouses, cities

def get_scrap_mask(
    sku_list: list[str],
    sku_info: pd.DataFrame,
    start_date = "2018-01-01"
) -> np.ndarray:
    """
    start_date 기준:
      1) (launch_date - start_date) > life_days  -> 출시 시점에 이미 수명 초과
      2) life - (launch - start_dt).days <= 30   -> 출시 후 30일 이내 수명 만료
    위 두 조건에 해당하는 SKU는 True(폐기대상) return
    """
    start_dt = pd.to_datetime(start_date)
    meta = sku_info.set_index("sku")
    mask = np.zeros(25, dtype = bool)
    sku_info = sku_info[["sku", "launch_date", "life_days"]]
    
    for i, sku in enumerate(sku_list):
        launch = meta.at[sku, "launch_date"]
        launch = pd.to_datetime(launch)
        life = int(meta.at[sku, "life_days"])
        cond1 = (launch - start_dt).days > life
        cond2 = life - (launch - start_dt).days <= 30
        mask[i] = bool(cond1 or cond2)
    
    return mask

def init_warehouse_stock(
    warehouses: dict[str, Warehouse],
    sku_list: list[str],
    sku_info: pd.DataFrame,
    tp_mode: dict[tuple[str, str], str],
    date = "2018-01-01"
) -> list[dict]:
    date_dt = pd.to_datetime(date)
    records: list[dict] = []

    # 폐기 대상 SKU만 추출하는 0/1 벡터
    scrap_mask = get_scrap_mask(sku_list, sku_info)

    for warehouse in warehouses.values():
        # 출하 벡터: 해당 SKU 전량
        ship_vector = (warehouse.stock * scrap_mask).astype(int)
        
        if not np.any(ship_vector):
            continue

        # 창고 재고 차감
        warehouse.stock -= ship_vector

        # 출하 기록 생성
        for sku, qty in zip(sku_list, ship_vector):
            if qty > 0:
                records.append({
                    "date": date,
                    "factory/warehouse": "warehouse",
                    "sku": sku,
                    "production_qty": 0,
                    "ot_qty": 0,
                    "ship_qty": int(qty),
                    "from": city_to_id(warehouse.city, "warehouse"),
                    "to": warehouse.city,
                    "mode": tp_mode[(warehouse.city, warehouse.city)]
                })

    return warehouses, records

# get the numbers of needed skus of a week using 'get_daily_warehouse_demand' function
def get_weekly_warehouse_demand(
    date: str, 
    warehouse: Warehouse, 
    leadtime: dict[tuple[str, str], int]
) -> np.ndarray:
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    weekly_demand = np.zeros(25, dtype = int)
    
    for n in range(7):
        obj_date = (date_dt + timedelta(days = n)).strftime("%Y-%m-%d")
        demand_dict = get_daily_warehouse_demand(obj_date, warehouse, leadtime)
        
        if demand_dict:
            weekly_demand += np.sum(list(demand_dict.values()), axis = 0).astype(int)
    
    return weekly_demand
        
# get the numbers of skus to be sent of the warehouse on the date
# that satisfies 95% fill-rate, not 100% of the demand
# Output: demand(dictionary) - {city: demand}
def get_daily_warehouse_demand(
    date: str, 
    warehouse: Warehouse, 
    leadtime: dict[tuple[str, str], int]
) -> dict[str, np.ndarray]:
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    demand: dict[str, np.ndarray] = {}
    
    for city in warehouse.client:
        obj_date = (date_dt + timedelta(days = leadtime[warehouse.city, city])).strftime("%Y-%m-%d")
        demand[city] = get_daily_city_demand(obj_date, city)
        
    return demand

# get the daily demand of the city 
# 2023-01-01 이전 발생한 수요의 경우, 예측값이 아닌 관측값이므로 95% fill-rate 만족하는 수준으로 보정
def get_daily_city_demand(
    date: str, 
    city: str
):
    return DEMAND_STORE.get(date, city)

# Update delivered stocks on the date
def daily_update(
    date: str, 
    warehouses: dict[str, Warehouse],
    cities: dict[str, City], 
):
    for warehouse in warehouses.keys():
        if date in warehouses[warehouse].dlv_in_prog:
            warehouses[warehouse].stock += warehouses[warehouse].dlv_in_prog[date]
            del warehouses[warehouse].dlv_in_prog[date]
    
    for city in cities.keys():
        if date in cities[city].dlv_in_prog:
            #cities[city].stock += cities[city].dlv_in_prog[date]
            del cities[city].dlv_in_prog[date]
    
    return warehouses, cities

# capa를 객체 내부에 클래스 변수로 선언해서 daily_update를 해주고 그 값을 이용할지, 그냥 여기서 불러서 사용할지 결정
# ot_capa 사용 시 처리 방안 만들어야함
def mps(
    date: str,
    factories: dict[str, Factory],
    warehouses: dict[str, Warehouse],
    sku_list: list[str],
    leadtime: dict[tuple[str, str], int],
    tp_mode: dict[tuple[str, str], str],
    orders: dict[str, dict[str, np.ndarray]],
    labour_path = "data/labour_requirement.csv",
):
    record = []
    wk = week_start(date)
    weekly_capacity = get_capacity(date)
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    labour_vector = pd.read_csv(labour_path)["labour_hours_per_unit"].to_numpy(dtype = float)
    
    for factory, order_within_factory in orders.items():
        if machine_failure_flag(date, factory):
            continue
        
        total_order_within_factory = np.sum(list(order_within_factory.values()), axis = 0).astype(int)
        total_labour = float(np.dot(total_order_within_factory, labour_vector))
        
        row = weekly_capacity.loc[weekly_capacity["factory"] == city_to_id(factory, "factory")]
        reg_capa, ot_capa = row.iloc[0]["reg_capacity"], row.iloc[0]["ot_capacity"]
        
        if total_labour <= reg_capa:
            reg_hour, ot_hour = total_labour, 0.0
            
        elif total_labour <= reg_capa + ot_capa: 
            reg_hour, ot_hour = float(reg_capa), total_labour - reg_capa
        
        # record_df 작성 후 db에 저장
        for warehouse, order in order_within_factory.items():
            arrival_date = (date_dt + timedelta(days = leadtime[factory, warehouse])).strftime("%Y-%m-%d")
            dlv = warehouses[warehouse].dlv_in_prog.get(arrival_date, np.zeros_like(order))
            warehouses[warehouse].dlv_in_prog[arrival_date] = dlv + order
            
            for sku, qty, total in zip(sku_list, order, total_order_within_factory):
                record.append({
                    "date": date,
                    "factory/warehouse": "factory",
                    "sku": sku,
                    "production_qty": total,
                    "ot_qty": 0,
                    "ship_qty": qty,
                    "from": city_to_id(factory, "factory"),
                    "to": city_to_id(warehouse, "warehouse"),
                    "mode": tp_mode[factory, warehouse],
                    "week": wk,                
                    "factory": factory,      
                    "warehouse": warehouse
                })
    
    return warehouses, record

def warehouse_order_plan(
    date: str,
    factories: dict[str, Factory],
    warehouses: dict[str, Warehouse], 
    cities: dict[str, City], 
    sku_list: list[str], 
    leadtime: dict[tuple[str, str], int], 
    tp_mode: dict[tuple[str, str], str], 
):
    orders = defaultdict(dict)
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    
    for warehouse in warehouses.values():
        for factory, supply_vector in warehouse.supplier.items():
            if machine_failure_flag(date, factory): 
                continue
                
            cover_date = (date_dt + timedelta(days = leadtime[factory, warehouse.city])).strftime("%Y-%m-%d")
            demand = get_weekly_warehouse_demand(cover_date, warehouse, leadtime)
            
            weeks_later = 1
            stock = warehouse.stock.copy()

            while machine_failure_flag((date_dt + timedelta(weeks = weeks_later)).strftime("%Y-%m-%d"), factory):
                extra_date = (date_dt + timedelta(weeks = weeks_later)).strftime("%Y-%m-%d")
                demand += get_weekly_warehouse_demand(extra_date, warehouse, leadtime)
                weeks_later += 1
            
            # date 기준 warehouse의 stock 정보는 city demand 소화한 이후임(Update 완료된 data)
            # 따라서 date로부터 하루 지난 날 ~ leadtime 하루 전 날까지의 수요만 구해주면 됨
            LT = leadtime[factory, warehouse.city]
            if LT > 1:
                for n in range(1, LT):
                    extra_date = (date_dt + timedelta(days = n)).strftime("%Y-%m-%d")
                    outbound_dict = get_daily_warehouse_demand(extra_date, warehouse, leadtime)
                    daily_demand = np.sum(list(outbound_dict.values()), axis = 0).astype(int)
                    daily_outbound = np.minimum(daily_demand, stock)
                    stock -= daily_outbound
            
            stock = np.maximum(stock, 0)
            net_demand = np.maximum(demand - stock, 0)
            order = net_demand * supply_vector
            
            if np.any(order):
                orders[factory][warehouse.city] = order
                
    warehouses, record = mps(date, factories, warehouses, sku_list, leadtime, tp_mode, orders)
    
    return warehouses, record
    
# 창고 -> 도시 배송 계획
# 입력받은 date 기준 각 창고별로 들어온 주문에 대해 fill-rate 95% 수준으로만 배송
# 각 창고에 대해 창고의 현재 보유 재고와 도시의 주문량 비교 후 더 작은 값이 배송량이 되도록 결정
def city_order_plan(
    date: str, 
    warehouses: dict[str, Warehouse], 
    cities: dict[str, City], 
    sku_list: list[str], 
    leadtime: dict[tuple[str, str], int], 
    tp_mode: dict[tuple[str, str], str], 
):
    record = []
    
    for warehouse in warehouses.values():
        order = get_daily_warehouse_demand(date, warehouse, leadtime)
        
        for city in warehouse.client:
            actual_supply = np.minimum(order[city], warehouse.stock)
            warehouse.stock -= actual_supply
            
            arrival_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days = leadtime[warehouse.city, city])).strftime("%Y-%m-%d")
            dlv = cities[city].dlv_in_prog.get(arrival_date, np.zeros_like(actual_supply))
            cities[city].dlv_in_prog[arrival_date] = dlv + actual_supply.copy()

            for sku, qty in zip(sku_list, actual_supply):
                record.append({
                    "date": date,
                    "factory/warehouse": "warehouse",
                    "sku": sku,
                    "production_qty": 0,
                    "ot_qty": 0,
                    "ship_qty": int(qty),
                    "from": city_to_id(warehouse.city, "warehouse"),
                    "to": city,
                    "mode": tp_mode[warehouse.city, city]
                })
    
    return warehouses, cities, record

def relevel_factory_weeks(records_df: pd.DataFrame) -> pd.DataFrame:
    # 필요 컬럼 체크
    req = {"date","week","factory","warehouse","sku","ship_qty"}
    if not req.issubset(set(records_df.columns)):
        raise ValueError(f"records_df missing cols: {sorted(req - set(records_df.columns))}")

    # 노동시간 벡터 로딩
    lab_df = pd.read_csv("data/labour_requirement.csv")
    lab_map = lab_df.set_index("sku")["labour_hours_per_unit"].to_dict()

    # 주별 정규/OT 캐파
    all_mondays = sorted(records_df["week"].dropna().unique().tolist())
    capa_rows = []
    for wk in all_mondays:
        cap = get_capacity(wk)  # 주 월요일 넘김
        for _, r in cap.iterrows():
            capa_rows.append({"week": wk, "factory_id": r["factory"], "reg_capacity": float(r["reg_capacity"]), "ot_capacity": float(r["ot_capacity"])})
    capa_df = pd.DataFrame(capa_rows)
    # factory_id ↔ 이름 매핑용 (city_to_id 재사용). records_df에는 이름이 있으므로 id로 합칠 필요 시 변환
    # 여기서는 이름 기준으로 재구성
    # id→이름 역매핑 유틸이 없다면 get_site_info 등에서 만든 테이블을 써도 좋음

    # (week,factory) labour 계산 함수
    def labour_of(group):
        # group: rows for a given week/factory (factory rows만)
        # ship_qty × labour_per_unit 합
        return float(sum(lab_map.get(s,0.0) * q for s,q in zip(group["sku"], group["ship_qty"])))

    # 팩토리 출하만 필터
    fac_df = records_df[records_df["factory/warehouse"] == "factory"].copy()
    wh_df  = records_df[records_df["factory/warehouse"] == "warehouse"].copy()

    # (week,factory)별 현재 labour
    fac_df["lab_per_unit"] = fac_df["sku"].map(lab_map).fillna(0.0)
    fac_df["lab"] = fac_df["lab_per_unit"] * fac_df["ship_qty"]

    # 주차 오름차순, 공장별 순회
    weeks_sorted = sorted(fac_df["week"].unique().tolist())
    factories = sorted(fac_df["factory"].unique().tolist())

    # 캐파 조회(고장=0 처리)
    def week_reg_capacity(wk: str, factory: str) -> float:
        # 월요일 고장이면 0
        if machine_failure_flag(wk, factory):
            return 0.0
        cap = get_capacity(wk)
        row = cap.loc[cap["factory"] == city_to_id(factory, "factory")]
        return float(row.iloc[0]["reg_capacity"]) if not row.empty else 0.0

    # 메인 루프: t 주의 초과분을 t-1, t-2…로 이동
    fac_df = fac_df.sort_values(["week","factory"]).reset_index(drop = True)

    for factory in factories:
        # 공장별만 뽑아서 작업(속도↑)
        sub = fac_df[fac_df["factory"] == factory].copy()
        # 주 목록
        f_weeks = sorted(sub["week"].unique().tolist())

        # 주별 현재 labour 캐시
        week_lab = {wk: float(sub[sub["week"] == wk]["lab"].sum()) for wk in f_weeks}
        week_reg = {wk: week_reg_capacity(wk, factory) for wk in f_weeks}

        for t_idx in range(len(f_weeks)):
            wk_t = f_weeks[t_idx]
            lab_t = week_lab.get(wk_t, 0.0)
            reg_t = week_reg.get(wk_t, 0.0)

            excess_lab = max(0.0, lab_t - reg_t)  # 정규 초과(OT 포함)
            if excess_lab <= 0.0:
                continue

            # t-1, t-2, ... 으로 이동
            for p_idx in range(t_idx-1, -1, -1):
                if excess_lab <= 0.0:
                    break
                wk_p = f_weeks[p_idx]
                if machine_failure_flag(wk_p, factory):
                    continue
                lab_p = week_lab.get(wk_p, 0.0)
                reg_p = week_reg.get(wk_p, 0.0)
                slack = max(0.0, reg_p - lab_p)
                if slack <= 0.0:
                    continue

                move_lab_allow = min(excess_lab, slack)

                # t주의 (factory→warehouse, sku) 라인들 중에서 비중에 따라 이동
                mask_t = (fac_df["factory"] == factory) & (fac_df["week"] == wk_t)
                rows_t = fac_df.loc[mask_t].copy()
                if rows_t.empty:
                    break
                weights = rows_t["lab"].to_numpy()
                if weights.sum() <= 0:
                    break
                ratio = weights / weights.sum()

                # 각 행별 이동할 노동시간 → 수량
                move_lab_each = move_lab_allow * ratio
                lab_per_unit = rows_t["lab_per_unit"].to_numpy()
                qty_from_lab = np.zeros_like(move_lab_each)
                nz = lab_per_unit > 0
                qty_from_lab[nz] = np.floor(move_lab_each[nz] / lab_per_unit[nz])

                # 현재 수량 한도
                cur_qty = rows_t["ship_qty"].to_numpy().astype(float)
                move_qty = np.minimum(qty_from_lab, cur_qty).astype(int)
                if move_qty.sum() <= 0:
                    continue

                # fac_df에 반영: t에서 줄이고, p로 같은 route/sku를 복제 추가
                idxs = rows_t.index.to_numpy()
                fac_df.loc[idxs, "ship_qty"] -= move_qty
                # 감소에 따른 lab 갱신
                fac_df.loc[idxs, "lab"] = fac_df.loc[idxs, "lab_per_unit"] * fac_df.loc[idxs, "ship_qty"]

                moved = rows_t.copy()
                moved["ship_qty"] = move_qty
                moved["lab"] = moved["lab_per_unit"] * moved["ship_qty"]
                moved["date"] = wk_p
                moved["week"] = wk_p

                # 0 qty는 버림
                moved = moved[moved["ship_qty"] > 0]
                if not moved.empty:
                    fac_df = pd.concat([fac_df, moved], ignore_index=True)

                # 캐시 업데이트
                delta_lab = float((moved["lab"]).sum())
                week_lab[wk_t] -= delta_lab
                week_lab[wk_p] += delta_lab
                excess_lab -= delta_lab

            # t 주가 남아있으면 그대로(OT 또는 Limit로 남음)

    # 음수/0 정리
    fac_df = fac_df[fac_df["ship_qty"] > 0].copy()

    # 원본 records_df에 병합
    out = pd.concat([wh_df, fac_df], ignore_index=True)
    # 타입 정리
    out["ship_qty"] = out["ship_qty"].astype(int)
    out["production_qty"] = out.get("production_qty", out["ship_qty"]).astype(int)
    out["ot_qty"] = out.get("ot_qty", 0).astype(int)
    return out

def allocate_ot_after_relevel(records_df: pd.DataFrame, labour_csv = "data/labour_requirement.csv",
                              eps: float = 1e-6) -> pd.DataFrame:
    """
    relevel_factory_weeks() 이후 실행.
    (week,factory)별 정규/OT 캐파 기준으로 ot_qty를 재계산하고,
    최종적으로 reg_labour < reg_capacity - eps 를 강제하도록 정규→OT 전환을 미세 조정한다.
    """
    df = records_df.copy()

    fac = df[df["factory/warehouse"] == "factory"].copy()
    if fac.empty:
        return df

    req = {"week", "factory", "sku", "ship_qty", "date"}
    missing = req - set(fac.columns)
    if missing:
        raise ValueError(f"[allocate_ot_after_relevel] missing cols: {sorted(missing)}")

    # labour per unit join
    lab = pd.read_csv(labour_csv)[["sku", "labour_hours_per_unit"]]
    fac = fac.merge(lab, on = "sku", how = "left")
    fac["labour_hours_per_unit"] = fac["labour_hours_per_unit"].fillna(0.0)

    fac["ship_qty"] = pd.to_numeric(fac["ship_qty"], errors = "coerce").fillna(0).astype(int)
    fac["ot_qty"] = pd.to_numeric(fac.get("ot_qty", 0), errors = "coerce").fillna(0).astype(int)

    out_chunks = []
    for (wk, f), g in fac.groupby(["week", "factory"], as_index=False):
        # 고장 주: OT 전환도 의미 없으니 0 유지(검증에서 위반 여부 판단)
        if machine_failure_flag(wk, f):
            g.loc[:, "ot_qty"] = 0
            out_chunks.append(g)
            continue

        # 캐파 로드
        cap = get_capacity(wk)
        row = cap.loc[cap["factory"] == city_to_id(f, "factory")]
        reg_cap = float(row.iloc[0]["reg_capacity"]) if not row.empty else 0.0
        ot_cap  = float(row.iloc[0]["ot_capacity"])  if not row.empty else 0.0

        per_unit = g["labour_hours_per_unit"].to_numpy()
        ship     = g["ship_qty"].to_numpy()
        ot_qty   = g["ot_qty"].to_numpy()

        # 1) 현재 노동량
        reg_qty = ship - ot_qty
        reg_lab = float(np.sum(reg_qty * per_unit))
        ot_lab  = float(np.sum(ot_qty * per_unit))
        total_lab = reg_lab + ot_lab

        # 1차 배분(기존): 총노동량 기준으로 OT를 먼저 할당
        #   - reg는 reg_cap 까지, 나머지를 ot_cap 한도 내에서 OT로
        if total_lab > reg_cap:
            need_ot_lab = total_lab - reg_cap
            allow_ot_lab = max(0.0, ot_cap - ot_lab)
            assign_lab = min(need_ot_lab, allow_ot_lab)

            if assign_lab > eps and total_lab > 0:
                # 가중 배분 → 정수 보정
                weight = ship * per_unit
                if weight.sum() <= 0:
                    pass
                else:
                    ratio = weight / weight.sum()
                    target_lab_each = assign_lab * ratio

                    add_ot = np.zeros_like(ot_qty, dtype=int)
                    nz = per_unit > 0
                    add_ot[nz] = np.floor(target_lab_each[nz] / per_unit[nz]).astype(int)

                    # 한도: ship-ot
                    add_ot = np.minimum(add_ot, ship - ot_qty)

                    used = float(np.sum(add_ot * per_unit))
                    remain = assign_lab - used
                    if remain > eps:
                        # 분수부 큰 순으로 1개씩 올림
                        frac = np.zeros_like(target_lab_each)
                        frac[nz] = target_lab_each[nz] / per_unit[nz] - add_ot[nz]
                        order = np.argsort(-frac)
                        for idx in order:
                            if remain + 1e-12 < per_unit[idx]:
                                continue
                            if ot_qty[idx] + add_ot[idx] >= ship[idx]:
                                continue
                            add_ot[idx] += 1
                            remain -= per_unit[idx]
                            if remain <= eps:
                                break

                    ot_qty = ot_qty + add_ot
                    reg_qty = ship - ot_qty
                    reg_lab = float(np.sum(reg_qty * per_unit))
                    ot_lab  = float(np.sum(ot_qty * per_unit))

        # 2) 미세 조정(신규): reg_lab을 **반드시** reg_cap - eps 미만으로
        if reg_lab >= reg_cap - eps:
            delta = reg_lab - (reg_cap - eps)   # 줄여야 하는 정규 노동량
            ot_slack = max(0.0, ot_cap - ot_lab)
            if ot_slack > eps:
                need = min(delta, ot_slack)

                # per_unit 큰 SKU부터 1개씩 정규→OT 전환
                # (최소 개수로 조건 만족시키기 위함)
                order = np.argsort(-per_unit)
                for idx in order:
                    if need <= eps:
                        break
                    # 전환 가능한 수량(= 현재 정규 수량)
                    can_move = int(reg_qty[idx])
                    if can_move <= 0 or per_unit[idx] <= 0:
                        continue
                    # 한 번에 몇 개? 남은 need를 labour_per_unit로 나눠 최소 개수
                    k = int(np.ceil((need + 1e-12) / per_unit[idx]))
                    k = min(k, can_move)

                    # OT 여유도 확인
                    max_by_ot = int(np.floor((ot_slack - (ot_lab - (ot_lab)) + need + 1e-12) / per_unit[idx]))  # 보수적
                    # 실은 위 라인은 과도하니, 그냥 k로 진행하고 이후 need/ot_slack으로 루프 제어

                    # 전환 실행
                    ot_qty[idx] += k
                    reg_qty[idx] -= k
                    reg_lab -= k * per_unit[idx]
                    ot_lab += k * per_unit[idx]
                    need -= k * per_unit[idx]

        # 결과 반영
        g.loc[:, "ot_qty"] = ot_qty.astype(int)
        out_chunks.append(g.drop(columns = []))

    fac2 = pd.concat(out_chunks, ignore_index = True)

    # 공장 외 레코드 병합
    others = df[df["factory/warehouse"] != "factory"].copy()
    fac2["ship_qty"] = fac2["ship_qty"].astype(int)
    fac2["ot_qty"] = fac2["ot_qty"].astype(int)

    return pd.concat([others, fac2], ignore_index = True)

def write_records_db(df: pd.DataFrame, db_path: str, table: str, chunksize: int = 2000):
    REQUIRED_DB_COLS = [
        "date",
        "factory/warehouse",
        "sku",
        "production_qty",
        "ot_qty",
        "ship_qty",
        "from",
        "to",
        "mode"
    ]
    
    out = df.loc[:, REQUIRED_DB_COLS].copy()
    
    with sqlite3.connect(db_path) as conn:
        out.to_sql(table, conn, if_exists = "append", index = False, method = "multi", chunksize = chunksize)

def simulation(wh_sol, fc_sol):
    # 기본 데이터 로드
    wh_info, fc_info = get_site_info()
    _, _, leadtime, tp_mode = get_tp_info(fc_info)
    sku_info = get_sku_info()
    city_list = fc_info["city"].tolist()
    buffer: list[dict] = []
    
    def flush_buffer():
        if not buffer:
            return
        
        df = pd.DataFrame(buffer)
        df.to_sql(
            TABLE_NAME, conn,
            if_exists = "append", index = False,
            method = "multi", chunksize = 2_000
        )
        
        buffer.clear()

    # SKU 순서 & 매핑
    sku_list = sku_info["sku"].tolist()
    sku_to_idx = {sku: idx for idx, sku in enumerate(sku_list)}
    
    # DEMAND_STORE 초기화
    global DEMAND_STORE
    DEMAND_STORE = DemandStore(sku_to_idx)
    DEMAND_STORE.load(
        db_path = "data/demand_train.db",
        csv_path = "data/forecast_submission_template.csv",
        start = "2018-01-01",
        end = "2024-12-31"
    )
    
    # 공장/창고/도시 객체 인스턴스 생성 및 초기 세팅
    factories, warehouses, cities = init_setting(wh_sol, fc_sol, city_list, sku_list, sku_to_idx)
    warehouses, init_record = init_warehouse_stock(warehouses, sku_list, sku_info, tp_mode)
    buffer.extend(init_record)
    
    # DB 연결
    conn = sqlite3.connect("data/plan_submission_template.db")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    TABLE_NAME = "plan_submission_template"

    START_DATE, END_DATE = pd.to_datetime("2018-01-01"), pd.to_datetime("2024-12-31")
    for date_dt in (START_DATE + timedelta(days = n)
                         for n in range((END_DATE - START_DATE).days + 1)):
        date = date_dt.strftime("%Y-%m-%d")
        
        warehouses, cities = daily_update(date, warehouses, cities)
        warehouses, cities, record_warehouse = city_order_plan(date, warehouses, cities, sku_list, leadtime, tp_mode)
        
        if record_warehouse: buffer.extend(record_warehouse)
        
        if date_dt.weekday() == 0 and date_dt >= pd.Timestamp("2018-01-08"):
            orders = defaultdict(dict)
            for w in warehouses.values():
                for f, supply_vector in w.supplier.items():
                    if machine_failure_flag(date, f):
                        continue
                    
                    cover_date = (date_dt + timedelta(days = leadtime[f, w.city])).strftime("%Y-%m-%d")
                    demand = get_weekly_warehouse_demand(cover_date, w, leadtime)

                    stock = w.stock.copy()
                    LT = leadtime[f, w.city]
                    if LT > 1:
                        for n2 in range(1, LT):
                            d2 = (date_dt + timedelta(days=n2)).strftime("%Y-%m-%d")
                            out_dict = get_daily_warehouse_demand(d2, w, leadtime)
                            day_need = np.sum(list(out_dict.values()), axis = 0).astype(int)
                            ship_vec = np.minimum(day_need, stock)
                            stock -= ship_vec

                    net = np.maximum(demand - np.maximum(stock, 0), 0)
                    order_vec = (net * supply_vector).astype(int)
                    if np.any(order_vec):
                        orders[f][w.city] = order_vec

            warehouses, record_factory = mps(date, factories, warehouses, sku_list, leadtime, tp_mode, orders)
            if record_factory: buffer.extend(record_factory)

    
    print("Simulation ended")
    print("Start writing db...")
    records_df = pd.DataFrame(buffer)
    records_df = relevel_factory_weeks(records_df)
    records_df = allocate_ot_after_relevel(records_df)
    write_records_db(records_df, db_path = "data/plan_submission_template.db", table = "plan_submission_template")
    conn.close()
    print("All process finished!!!")
    
    return 0