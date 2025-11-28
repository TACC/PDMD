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

atomic_energy_map = {
    1: -0.5799280498,
    8: -15.9556439593
}

def generate_soap_force(number, pos, neighborlist=None):
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

    # for training and benchmarking when a neighborlist is not applicable
    if (neighborlist is None):
      system = Atoms(symbols=element_string, positions=pos)
      for i in range(len(element_array)):
        soap_descriptors = torch.from_numpy(soap.create(system, centers=[i],n_jobs=-1))
        one_hot_encoded = torch.zeros(2)
        if number[i] == 1:
            one_hot_encoded = torch.tensor([1, 0])
        if number[i] == 8:
            one_hot_encoded = torch.tensor([0, 1])
        soap_descriptors = torch.hstack((one_hot_encoded, soap_descriptors))
        soap_fea.append(soap_descriptors)
    # for MD simulations accelerated by a neighborlist
    else:
      system = Atoms(symbols=element_string, positions=pos)
      for i in range(len(element_array)):
        nl_indices, nl_offsets = neighborlist.get_neighbors(i)
        subsystem = system[nl_indices]
        subsystem_center = nl_indices.tolist().index(i)
        soap_descriptors = torch.from_numpy(soap.create(subsystem, centers=[subsystem_center],n_jobs=-1))
        one_hot_encoded = torch.zeros(2)
        if number[i] == 1:
            one_hot_encoded = torch.tensor([1, 0])
        if number[i] == 8:
            one_hot_encoded = torch.tensor([0, 1])
        soap_descriptors = torch.hstack((one_hot_encoded, soap_descriptors))
        soap_fea.append(soap_descriptors)

    return soap_fea

def one_time_generate_forward_input_force(number, pos, CMA, forces_feature_min_values, forces_feature_max_values, neighborlist=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_full = generate_soap_force(number, pos, neighborlist)
    x_full = torch.stack(x_full).to(device)
    x_full = x_full.to(torch.float32)
    forces_feature_min_values = forces_feature_min_values.to(device).to(torch.float32)
    forces_feature_max_values = forces_feature_max_values.to(device).to(torch.float32)

    x_full = (x_full - forces_feature_min_values) / (forces_feature_max_values - forces_feature_min_values)
    # x_full = torch.stack(x_full)
    # for i in range(722):
    #     x_full[:, i] = tensor_min_max_scaler_1d(x_full[:, i], max=max_value[i], min=min_value[i])
    # x_full = x_full.to(torch.float32).to(device)

    pos = pos.to(device)
    DMA = torch.cdist(pos, pos)
    BTMA = torch.zeros_like(DMA, dtype=int)

    if CMA.numel() == 0:
     # Adaptive Cutoff
     print("Initializaiton Force CMA...")
     CMA = torch.zeros_like(DMA)
     cutoffs = {(1, 1): 1.6, (1, 8): 2.4, (8, 1): 2.4, (8, 8): 2.8}
     for i, atom_i in enumerate(number):
        for j, atom_j in enumerate(number):
            CMA[i,j] = cutoffs[(atom_i, atom_j)]
     
    BTMA = torch.where((DMA-CMA)<=0.0, 1, 0)

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
    return x, CMA

def generate_soap_energy(number, pos, neighborlist=None):
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

    # for training and benchmarking when a neighborlist is not applicable
    if (neighborlist is None):
      system = Atoms(symbols=element_string, positions=pos)
      tem = []
      for iatom in range(len(element_array)):
        output_stream = io.StringIO()
        with contextlib.redirect_stdout(output_stream):
            soap_descriptors = torch.from_numpy(soap.create(system, centers=[iatom],n_jobs=-1))
            one_hot_encoded = torch.zeros(2)
            if number[iatom] == 1:
                one_hot_encoded = torch.tensor([1, 0])
            if number[iatom] == 8:
                one_hot_encoded = torch.tensor([0, 1])
            soap_descriptors = torch.hstack((one_hot_encoded, soap_descriptors))
        tem.append(soap_descriptors)
    # for MD simulations accelerated by a neighborlist
    else:
      system = Atoms(symbols=element_string, positions=pos)
      tem = []
      for iatom in range(len(element_array)):
        output_stream = io.StringIO()
        with contextlib.redirect_stdout(output_stream):
            nl_indices, nl_offsets = neighborlist.get_neighbors(iatom)
            subsystem = system[nl_indices]
            subsystem_center = nl_indices.tolist().index(iatom) 
            soap_descriptors = torch.from_numpy(soap.create(subsystem, centers=[subsystem_center],n_jobs=-1))
            one_hot_encoded = torch.zeros(2)
            if number[iatom] == 1:
                one_hot_encoded = torch.tensor([1, 0])
            if number[iatom] == 8:
                one_hot_encoded = torch.tensor([0, 1])
            soap_descriptors = torch.hstack((one_hot_encoded, soap_descriptors))
        tem.append(soap_descriptors)

    return tem

def one_time_generate_forward_input_energy(number, pos, CMA, energy_feature_min_values, energy_feature_max_values, neighborlist=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_full = generate_soap_energy(number, pos, neighborlist)
    x_full = torch.stack(x_full).to(device)
    x_full = x_full.to(torch.float32)
    energy_feature_min_values = energy_feature_min_values.to(device).to(torch.float32)
    energy_feature_max_values = energy_feature_max_values.to(device).to(torch.float32)

    x_full = (x_full - energy_feature_min_values) / (energy_feature_max_values - energy_feature_min_values)

    pos = pos.to(device)
    DMA = torch.cdist(pos, pos)
    BTMA = torch.zeros_like(DMA, dtype=torch.int, device=device)

    if CMA.numel() == 0:
     # Adaptive Cutoff
     print("Initializaiton Force CMA...")
     CMA = torch.zeros_like(DMA)
     cutoffs = {(1, 1): 1.6, (1, 8): 2.4, (8, 1): 2.4, (8, 8): 2.8}
     for i, atom_i in enumerate(number):
        for j, atom_j in enumerate(number):
            CMA[i,j] = cutoffs[(atom_i, atom_j)]

    BTMA = torch.where((DMA-CMA)<=0.0, 1, 0)

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

    return x, CMA

def reverse_min_max_scaler(data_normalized, data_min=-361.77515914, data_max=-17.19880547, new_min=0.0, new_max=1.0):
    core = (data_normalized - new_min) / (new_max - new_min)
    data_original = core * (data_max - data_min) + data_min
    return data_original

def molecular_energy(atomic_numbers, energy, atomic_energy_map):
    for atom in atomic_numbers:
        if atom in atomic_energy_map:
            energy += atomic_energy_map[atom]
        else:
            print(f"Warning: Atomic number {atom} not found in the atomic energy map!")
    return energy

