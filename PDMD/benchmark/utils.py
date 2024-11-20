import torch
import math
from dscribe.descriptors import SOAP
from ase import Atoms
from torch_geometric.utils import degree
import io
import contextlib
import numpy as np

def tensor_min_max_scaler_1d(data, max, min, new_min=0.0, new_max=1.0):
    assert torch.is_tensor(data)
    data_min = min
    data_max = max
    assert data_max - data_min > 0
    core = (data - data_min) / (data_max - data_min)
    data_new = core * (new_max - new_min) + new_min
    return data_new

element_map = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca"
}

def generate_soap_force(number, pos):
    soap_fea = []
    element_array = [element_map[num] for num in number]
    element_string = ''.join(element_array)
    soap = SOAP(
        species=set(number),
        r_cut=10.0,
        n_max=10,
        l_max=5,
        periodic=False,
        average="cc"
    )
    system = Atoms(symbols=element_string, positions=pos)
    for i in range(len(element_array)):
        soap_descriptors = torch.from_numpy(soap.create(system, centers=[i]))
        one_hot_encoded = torch.zeros(2)
        if number[i] == 1:
            one_hot_encoded = torch.tensor([1, 0])
        if number[i] == 8:
            one_hot_encoded = torch.tensor([0, 1])
        soap_descriptors = torch.hstack((one_hot_encoded, soap_descriptors))
        soap_fea.append(soap_descriptors)
    return soap_fea

def one_time_generate_forward_input_force(number, pos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_value = []
    min_value = []
    with open("./PDMD/benchmark/max_values_force_onehot_expand.txt", "r") as max_file:
        for line in max_file:
            max_value.append(float(line.strip()))
    with open("./PDMD/benchmark/min_values_force_onehot_expand.txt", "r") as min_file:
        for line in min_file:
            min_value.append(float(line.strip()))

    x_full = generate_soap_force(number, pos)
    x_full = torch.stack(x_full).to(device)
    x_full = x_full.to(torch.float32)
    max_values = torch.tensor(max_value, dtype=torch.float32, device=device)
    min_values = torch.tensor(min_value, dtype=torch.float32, device=device)
    x_full = (x_full - min_values) / (max_values - min_values)
    # x_full = torch.stack(x_full)
    # for i in range(722):
    #     x_full[:, i] = tensor_min_max_scaler_1d(x_full[:, i], max=max_value[i], min=min_value[i])
    # x_full = x_full.to(torch.float32).to(device)

    pos = pos.to(device)
    DMA = torch.cdist(pos, pos)
    BTMA = torch.zeros_like(DMA, dtype=int)
    # Adaptive Cutoff
    cutoffs = {(1, 1): 1.6, (1, 8): 2.4, (8, 1): 2.4, (8, 8): 2.8}
    for i, atom_i in enumerate(number):
        for j, atom_j in enumerate(number):
            cutoff = cutoffs[(atom_i, atom_j)]
            if DMA[i, j] <= cutoff:
                BTMA[i, j] = 1
    """
    mask = DMA <= 1.5
    BTMA[mask] = 1
    """
    BTMA.fill_diagonal_(0)
    adj = DMA * BTMA
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
    c = int(pos.shape[0])
    batch = []
    for i in range(int(c / len(number))):
        batch += [i] * len(number)
    batch = torch.tensor(batch).to(device)
    x = dict({
        "x": x_full,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch
    })
    return x

def generate_soap_energy(number, pos):
    element_array = [element_map[num] for num in number]
    element_string = ''.join(element_array)
    soap = SOAP(
        species=set(number),
        r_cut=10.0,
        n_max=10,
        l_max=5,
        periodic=False,
        average="outer"
    )

    system = Atoms(symbols=element_string, positions=pos)
    tem = []
    for iatom in range(len(element_array)):
        output_stream = io.StringIO()
        with contextlib.redirect_stdout(output_stream):
            soap_descriptors = torch.from_numpy(soap.create(system, centers=[iatom]))
            one_hot_encoded = torch.zeros(2)
            if number[iatom] == 1:
                one_hot_encoded = torch.tensor([1, 0])
            if number[iatom] == 8:
                one_hot_encoded = torch.tensor([0, 1])
            soap_descriptors = torch.hstack((one_hot_encoded, soap_descriptors))
        tem.append(soap_descriptors)
    return tem

def one_time_generate_forward_input_energy(number, pos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_value = []
    min_value = []
    with open("./PDMD/benchmark/max_values_energy_onehot_expand.txt", "r") as max_file:
        for line in max_file:
            max_value.append(float(line.strip()))
    with open("./PDMD/benchmark/min_values_energy_onehot_expand.txt", "r") as min_file:
        for line in min_file:
            min_value.append(float(line.strip()))

    x_full = generate_soap_energy(number, pos)
    x_full = torch.stack(x_full).to(device)
    x_full = x_full.to(torch.float32)
    max_values = torch.tensor(max_value, dtype=torch.float32, device=device)
    min_values = torch.tensor(min_value, dtype=torch.float32, device=device)
    x_full = (x_full - min_values) / (max_values - min_values)

    pos = pos.to(device)
    DMA = torch.cdist(pos, pos)
    BTMA = torch.zeros_like(DMA, dtype=torch.int, device=device)

    # Adaptive Cutoff
    cutoffs = {(1, 1): 1.6, (1, 8): 2.4, (8, 1): 2.4, (8, 8): 2.8}
    for i, atom_i in enumerate(number):
        for j, atom_j in enumerate(number):
            cutoff = cutoffs[(atom_i, atom_j)]
            if DMA[i, j] <= cutoff:
                BTMA[i, j] = 1
    """
    mask = DMA <= 1.5
    BTMA[mask] = 1
    """
    BTMA.fill_diagonal_(0)
    adj = DMA * BTMA
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
    c = int(pos.shape[0])
    batch = []
    for i in range(int(c / len(number))):
        batch += [i] * len(number)
    batch = torch.tensor(batch).to(device)

    x = dict({
        "x": x_full,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch
    })
    return x

def reverse_min_max_scaler(data_normalized, data_min=-361.77515914, data_max=-17.19880547, new_min=0.0, new_max=1.0):
    core = (data_normalized - new_min) / (new_max - new_min)
    data_original = core * (data_max - data_min) + data_min
    return data_original

