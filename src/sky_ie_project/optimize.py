import gurobipy as gp
from io_utils import *

def wh_optimize():
    model = gp.Model("Warehouse_Location_Optimization")
    
    # Load data
    wh_info, fc_info = get_site_info()
    sku_info = get_sku_info()
    daily_demand = get_avg_demand(fc_info)
    tp_cost, tp_carbon, _, _ = get_tp_info(fc_info)
    
    # Preprocess data for optimization model
    cities = fc_info["city"].tolist()  # 도시 리스트
    skus = sku_info["sku"].tolist()  # SKU 리스트
    wh_cost = dict(zip(wh_info["city"], wh_info["init_cost_usd"])) # 창고 설치 비용
    
    # Define sets
    W = cities # 창고 후보지
    C = cities # 도시 리스트
    U = skus # SKU 리스트
    
    # Define decision variables
    ## Binary decision variables
    N = model.addVars(W, vtype = gp.GRB.BINARY, name = "N")  # Decide whether to build a warehouse in city W
    Y = model.addVars(W, C, vtype = gp.GRB.BINARY, name = "Y") # Decide whether to deliver goods from warehouse W to city C
    
    ## Non-negative integer decision variables
    Q = model.addVars(W, C, U, vtype = gp.GRB.INTEGER, name = "Q")  # Amount of sku S delivered from warehouse W to city C
    S = model.addVars(W, C, vtype = gp.GRB.INTEGER, name = "S")  # Amount of containers deliverd from warehouse W to city C
    T = model.addVar(vtype = gp.GRB.INTEGER, name = "T")  # Amount of CO2 emissions (Round up integer)

    # Parameters for better readability
    total_carbon = gp.quicksum(tp_carbon[j, k] * S[j, k] for j in W for k in C)
    
    # Objective function: minimization of total cost
    model.setObjective(
        gp.quicksum(N[j] * wh_cost[j] / (365 * 7) for j in W) + # 창고 설치비용 -> 일간 단위로 나누기
        gp.quicksum(Y[j, k] * tp_cost[j, k] for j in W for k in C) + # 배송비용 (창고 -> 도시)
        200 * T, # 탄소배출비용
        gp.GRB.MINIMIZE)
    
    # Constraints    
    model.addConstr(
        gp.quicksum(N[j] for j in W) <= 20, 
        name = "max_wh_limit")  # 창고 최대 개수 제약
    model.addConstrs(
        (Y[j, k] <= N[j] for j in W for k in C), 
        name = "edge_Y_constraints")  # 창고에서 도시로의 배송은 창고가 설치된 경우에만 가능
    model.addConstrs(
        (gp.quicksum(Y[j, k] for j in W) >= 1 for k in C), 
        name = "city_inbound_constraints")  # 모든 도시는 적어도 하나의 창고에서 물건을 받아야 함
    model.addConstrs(
        (gp.quicksum(Y[j, k] for k in C) >= N[j] for j in W), 
        name = "wh_outbound_constraints")  # 설치된 모든 창고는 적어도 하나의 도시에 물건을 보내야 함
    model.addConstrs(
        (Q[j, k, u] <= 20000 * Y[j, k] for j in W for k in C for u in U), 
        name = "edge_Y_tp_constraints")  # 창고-도시 엣지 연결 시에만 배송 가능
    model.addConstrs(
        (gp.quicksum(Q[j, k, u] for j in W) == daily_demand[k, u] for k in C for u in U), 
        name = "tp_demand_conditions")  # 도시별 각 SKU의 일간 수요는 배송량과 동일해야 함
    model.addConstrs(
        (gp.quicksum(Q[j, k, u] for u in U) <= 4000 * S[j, k] for j in W for k in C), 
        name = "wh_container_lower_limit")  # 창고-도시 컨테이너 선형화
    model.addConstrs(
        (gp.quicksum(Q[j, k, u] for u in U) + 3999 >= 4000 * S[j, k] for j in W for k in C), 
        name = "wh_container_upper_limit")  # 창고-도시 컨테이너 선형화(가지치기 제약)
    model.addConstr(
        total_carbon <= 1000 * T, 
        name = "carbon_emission_lower_limit")  # 탄소배출비용 선형화
    model.addConstr(
        total_carbon + 999 >= 1000 * T, 
        name = "carbon_emission_upper_limit")  # 탄소배출비용 선형화(가지치기 제약)
    
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found\n")
        print("Objective function value:", model.ObjVal)
        solution = {
            "N": {j: N[j].X for j in W if N[j].X > 0},
            "Y": {(j, k): Y[j, k].X for j in W for k in C if Y[j, k].X > 0}
        }
        return solution
    
    else:
        print("No optimal solution found")

def fc_optimize(wh_sol):
    model = gp.Model("Factory_Location_Optimization")
    
    # Load data
    wh_info, fc_info = get_site_info()
    sku_info = get_sku_info()
    daily_demand = get_avg_demand(fc_info)
    mt_cost = get_mt_cost(fc_info)
    wage = get_wage(fc_info)
    tp_cost, tp_carbon, _, _ = get_tp_info(fc_info)
    
    # Preprocess data for optimization model
    cities = fc_info["city"].tolist()  # 도시 리스트
    skus = sku_info["sku"].tolist()  # SKU 리스트
    fc_cost = dict(zip(fc_info["city"], fc_info["init_cost_usd"])) # 공장 설치 비용
    labour_hour = dict(zip(sku_info["sku"], sku_info["labour_hours_per_unit"]))  # SKU별 노동 시간 정보
    fc_carbon = dict(zip(fc_info["city"], fc_info["kg_CO2_per_unit"]))  # 공장별 탄소 배출 계수
    wh_loc, wh_ct_edge = get_wh_info(wh_sol) # 선택된 창고 node 및 창고-도시 edge 정보
    
    # Define sets
    F = cities # 공장 후보지
    W = sorted(wh_loc) # 설치된 창고
    U = skus # SKU 리스트
    
    demand_wh_u = aggregate_demand_by_wh(wh_loc, wh_ct_edge, daily_demand, U) # 창고별 수요 집계
    # Define decision variables
    ## Binary decision variables
    M = model.addVars(F, vtype = gp.GRB.BINARY, name = "M")  # Decide whether to build a factory in city F
    X = model.addVars(F, W, vtype = gp.GRB.BINARY, name = "X") # Decide whether to transport goods from factory F to warehouse W
    
    ## Non-negative integer decision variables
    P = model.addVars(F, W, U, vtype = gp.GRB.INTEGER, name = "P")  # Amount of sku S produced at factory F and sent to warehouse W
    R = model.addVars(F, W, vtype = gp.GRB.INTEGER, name = "R")  # Amount of containers deliverd from factory F to warehouse W
    T = model.addVar(vtype = gp.GRB.INTEGER, name = "T")  # Amount of CO2 emissions (Round up integer)

    # Parameters for better readability
    rho = {(i, u): gp.quicksum(P[i, j, u] for j in W) for i in F for u in U} # 공장 i에서 만든 SKU u의 양
    total_carbon = (
        gp.quicksum(fc_carbon[i] * gp.quicksum(rho[i, u] for u in U) for i in F) +
        gp.quicksum(tp_carbon[(i, j)] * R[i, j] for i in F for j in W)
    )
    
    # Objective function: minimization of total cost
    model.setObjective(
        gp.quicksum(M[i] * fc_cost[i] / (52 * 7) for i in F) + # 공장 설치비용 -> 주간 단위로 나누기
        gp.quicksum(rho[i, u] * (mt_cost[i, u] + wage[i] * labour_hour[u]) for i in F for u in U) + # 생산비용
        gp.quicksum(X[i, j] * tp_cost[i, j] for i in F for j in W) + # 운송비용 (공장 -> 창고)
        200 * T, # 탄소배출비용
        gp.GRB.MINIMIZE)
    
    # Constraints    
    model.addConstr(
        gp.quicksum(M[i] for i in F) <= 5, 
        name = "max_fc_limit")  # 공장 최대 개수 제약
    model.addConstrs(
        (X[i, j] <= M[i] for i in F for j in W), 
        name = "edge_X_constraints")  # 공장에서 창고로의 운송은 공장이 설치된 경우에만 가능
    model.addConstrs(
        (gp.quicksum(X[i, j] for j in W) >= M[i] for i in F), 
        name = "fc_outbound_constraints")  # 설치된 모든 공장은 적어도 하나의 창고에 물건을 보내야 함
    model.addConstrs(
        (gp.quicksum(X[i, j] for i in F) >= 1 for j in W), 
        name = "wh_inbound_constraints")  # 설치된 모든 창고는 적어도 하나의 공장에서 물건을 받아야 함
    model.addConstrs(
        (P[i, j, u] <= 40000 * X[i, j] for i in F for j in W for u in U), 
        name = "edge_X_tp_constraints")  # 공장-창고 엣지 연결 시에만 운송 가능
    model.addConstrs(
        (gp.quicksum(P[i, j, u] for i in F) == 7 * demand_wh_u[(j, u)] for j in W for u in U), 
        name = "inbound_outbound_conditions")  # 공장의 생산량은 창고의 수요량과 같아야 함
    model.addConstrs(
        (gp.quicksum(P[i, j, u] for u in U) <= 4000 * R[i, j] for i in F for j in W), 
        name = "fc_container_lower_limit")  # 공장-창고 컨테이너 선형화
    model.addConstrs(
        (gp.quicksum(P[i, j, u] for u in U) + 3999 >= 4000 * R[i, j] for i in F for j in W), 
        name = "fc_container_upper_limit")  # 공장-창고 컨테이너 선형화(가지치기 제약)
    model.addConstr(
        total_carbon <= 1000 * T, 
        name = "carbon_emission_lower_limit")  # 탄소배출비용 선형화
    model.addConstr(
        total_carbon + 999 >= 1000 * T, 
        name = "carbon_emission_upper_limit")  # 탄소배출비용 선형화(가지치기 제약)
    
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        solution = {
            "M": {i: M[i].X for i in F if M[i].X > 0},
            "X": {(i, j): X[i, j].X for i in F for j in W if X[i, j].X > 0},
            "P": {(i, j, u): P[i, j, u].X for i in F for j in W for u in U if P[i, j, u].X > 0}
        }
        return solution
