from dataclasses import dataclass, field
import numpy as np

@dataclass(slots = True)
class Factory: # Define factory object
    city:           str
    open_date:      str         # "YYYY-MM-DD"

@dataclass(slots = True)
class Warehouse: # Define warehouse object
    city:           str
    open_date:      str         # "YYYY-MM-DD"
    client:         list[str]   # 연결된 도시의 도시명
    supplier:       dict[str, np.ndarray] = field(default_factory = lambda: {}) # factory(city): binary vector(sku)
    stock:          np.ndarray = field(default_factory = lambda: np.zeros(25, dtype = int)) # vector info about the amount of each sku
    dlv_in_prog:    dict[str, np.ndarray] = field(default_factory = dict) # delivery in progress {date: vector(sku)}

@dataclass(slots = True)
class City: # Define City object
    city:           str
    dlv_in_prog:    dict[str, np.ndarray] = field(default_factory = dict)
