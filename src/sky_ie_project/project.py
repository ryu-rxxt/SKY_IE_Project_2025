from flag import *
from domain import *
from io_utils import *
from optimize import *
from simulation import *

def main():
    wh_sol = wh_optimize()
    fc_sol = fc_optimize(wh_sol)
    simulation(wh_sol, fc_sol)

if __name__ == "__main__":
    main()