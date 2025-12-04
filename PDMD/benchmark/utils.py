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

def generate_soap_force(number, pos, neighborlist_soap=None):
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
    if (neighborlist_soap is None):
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
      
      subsystems = []
      subsystem_centers = [] 

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      box = float('inf') * torch.eye(3,dtype=torch.float64).to(device)
      points = pos.to(device)
      nl_ij = neighborlist_soap.compute(points=points,box=box,periodic=False,quantities="P")[0]
      for i in range(len(element_array)):
        nl_indices = nl_ij[nl_ij[:,0] == i,1].tolist()
        nl_indices.insert(0,i)
        subsystems.append(system[nl_indices])
        subsystem_centers.append([0])

      soap_descriptors = torch.from_numpy(soap.create(subsystems, centers=subsystem_centers,n_jobs=-1))
     
      for i in range(len(element_array)):
        one_hot_encoded = torch.zeros(2)
        if number[i] == 1:
            one_hot_encoded = torch.tensor([1, 0])
        if number[i] == 8:
            one_hot_encoded = torch.tensor([0, 1])
        soap_fea.append(torch.hstack((one_hot_encoded, soap_descriptors[i])))
       
      del subsystems, subsystem_centers

    return soap_fea

def one_time_generate_forward_input_force(number, pos, forces_feature_min_values, forces_feature_max_values, neighborlist_soap=None, neighborlist_chemgnn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_full = generate_soap_force(number, pos, neighborlist_soap)
    x_full = torch.stack(x_full).to(device)
    x_full = x_full.to(torch.float32)
    forces_feature_min_values = forces_feature_min_values.to(device).to(torch.float32)
    forces_feature_max_values = forces_feature_max_values.to(device).to(torch.float32)

    x_full = (x_full - forces_feature_min_values) / (forces_feature_max_values - forces_feature_min_values)

    pos = pos.to(device)

    cutoffs = {(1, 1): 1.6, (1, 8): 2.4, (8, 1): 2.4, (8, 8): 2.8}
    if (neighborlist_chemgnn is None):
       print("A neighbor list is required for MD runs!") 
    else:
       edge_index = []
       edge_attr = []

       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       box = float('inf') * torch.eye(3,dtype=torch.float64).to(device)
       points = pos.to(device)
       nl_ij, nl_d = neighborlist_chemgnn.compute(points=points,box=box,periodic=False,quantities="Pd")

       for i, atom_i in enumerate(number):
          nl_indices = nl_ij[nl_ij[:,0] == i,1].tolist()
          nl_d_i = nl_d[nl_ij[:,0] == i].tolist()
          for j_d, j in enumerate(nl_indices):
              atom_j = number[j]
              distance_ij = nl_d_i[j_d]
              if ( distance_ij < cutoffs[(atom_i, atom_j)]):
                  edge_index.append([i,j])
                  edge_attr.append(distance_ij)

       edge_index = torch.tensor(edge_index).t().to(device)
       edge_attr = torch.tensor(edge_attr).to(torch.long).to(device)

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

def generate_soap_energy(number, pos, neighborlist_soap=None):
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
    if (neighborlist_soap is None):
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
      
      subsystems = []
      subsystem_centers = []

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      box = float('inf') * torch.eye(3,dtype=torch.float64).to(device)
      points = pos.to(device)
      nl_ij = neighborlist_soap.compute(points=points,box=box,periodic=False,quantities="P")[0]
      for iatom in range(len(element_array)):
         nl_indices = nl_ij[nl_ij[:,0] == iatom,1].tolist()
         nl_indices.insert(0,iatom)
         subsystems.append(system[nl_indices])
         subsystem_centers.append([0])

      output_stream = io.StringIO()
      with contextlib.redirect_stdout(output_stream):
         soap_descriptors = torch.from_numpy(soap.create(subsystems, centers=subsystem_centers,n_jobs=-1))
     
      for iatom in range(len(element_array)):                           
         one_hot_encoded = torch.zeros(2)
         if number[iatom] == 1:
             one_hot_encoded = torch.tensor([1, 0])
         if number[iatom] == 8:
             one_hot_encoded = torch.tensor([0, 1])
         tem.append(torch.hstack((one_hot_encoded, soap_descriptors[iatom])))

      del subsystems, subsystem_centers

    return tem

def one_time_generate_forward_input_energy(number, pos, energy_feature_min_values, energy_feature_max_values, neighborlist_soap=None, neighborlist_chemgnn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    x_full = generate_soap_energy(number, pos, neighborlist_soap)

    x_full = torch.stack(x_full).to(device)
    x_full = x_full.to(torch.float32)
    energy_feature_min_values = energy_feature_min_values.to(device).to(torch.float32)
    energy_feature_max_values = energy_feature_max_values.to(device).to(torch.float32)

    x_full = (x_full - energy_feature_min_values) / (energy_feature_max_values - energy_feature_min_values)

    pos = pos.to(device)

    cutoffs = {(1, 1): 1.6, (1, 8): 2.4, (8, 1): 2.4, (8, 8): 2.8}
    if (neighborlist_chemgnn is None):
       print("A neighbor list is required for MD runs!")
    else:
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       box = float('inf') * torch.eye(3,dtype=torch.float64).to(device)
       points = pos.to(device)
       nl_ij, nl_d = neighborlist_chemgnn.compute(points=points,box=box,periodic=False,quantities="Pd")
 
       edge_index = []
       edge_attr = []
       for i, atom_i in enumerate(number):
          nl_indices = nl_ij[nl_ij[:,0] == i,1].tolist()
          nl_d_i = nl_d[nl_ij[:,0] == i].tolist()
          for j_d, j in enumerate(nl_indices):
              atom_j = number[j]
              distance_ij = nl_d_i[j_d]
              if ( distance_ij < cutoffs[(atom_i, atom_j)]):
                  edge_index.append([i,j])
                  edge_attr.append(distance_ij)

       edge_index = torch.tensor(edge_index).t().to(device)
       edge_attr = torch.tensor(edge_attr).to(torch.long).to(device)

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

def molecular_energy(atomic_numbers, energy, atomic_energy_map):
    for atom in atomic_numbers:
        if atom in atomic_energy_map:
            energy += atomic_energy_map[atom]
        else:
            print(f"Warning: Atomic number {atom} not found in the atomic energy map!")
    return energy

