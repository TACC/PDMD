import torch
import math
from dscribe.descriptors import SOAP
from ase import Atoms
from torch_geometric.utils import degree
import io
import contextlib
import numpy as np
from torch_scatter import scatter

def compute_angles(pos: torch.Tensor, idx: torch.Tensor,
                       eps: float = 1e-12, to_degrees: bool = False
                      ) -> torch.Tensor:
    """
    Compute bond angles for a list of atom triplets.

    Parameters
    ----------
    pos : torch.Tensor, shape (N, 3)
        Cartesian coordinates of the N atoms (float32 or float64).
    idx : torch.Tensor, shape (3, M)
        For each row (i, j, k) the angle ∠i‑j‑k is computed,
        i.e. the angle at atom j between the vectors  i→j and k→j.
    eps : float, optional
        Small value added to the denominator for numerical stability.
    to_degrees : bool, optional
        If True, return angles in degrees; otherwise in radians (default).

    Returns
    -------
    angles : torch.Tensor, shape (M,)
        The computed angles (one per triplet).
    """
    pos_i = pos[idx[0, :]]
    pos_j = pos[idx[1, :]]
    pos_k = pos[idx[2, :]]
    v_ij = pos_i - pos_j
    v_kj = pos_k - pos_j
    norm_ij = torch.norm(v_ij, dim=1, keepdim=True).clamp(min=eps)
    norm_kj = torch.norm(v_kj, dim=1, keepdim=True).clamp(min=eps)
    v_ij_norm = v_ij / norm_ij
    v_kj_norm = v_kj / norm_kj
    cos_angle = (v_ij_norm * v_kj_norm).sum(dim=1)
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    if to_degrees:
        angle_rad = angle_rad * 180.0 / torch.pi

    return angle_rad

def edges_to_angles(edges: torch.Tensor) -> torch.Tensor:
    """Build the ``3 x M`` hyperedge tensor from a ``2 x N`` edge tensor.

    Parameters
    ----------
    edges : Tensor
        ``(2, N)`` integer tensor of node-id pairs. May live on CPU or CUDA.

    Returns
    -------
    Tensor
        ``(3, M)`` tensor on the same device/dtype as ``edges``; middle row is
        the shared node, columns sorted by ``(c, a, b)``, all columns unique.
    """
    if edges.dim() != 2 or edges.size(0) != 2:
        raise ValueError(f"`edges` must have shape (2, N); got {tuple(edges.shape)}")

    device, dtype = edges.device, edges.dtype
    empty = torch.empty((3, 0), device=device, dtype=dtype)

    if edges.size(1) == 0:
        return empty

    u, v = edges[0], edges[1]

    # 1. Expand each undirected edge {u, v} into both incidences (center, other).
    center = torch.cat((u, v))
    other = torch.cat((v, u))

    # 2. Drop self-loops: a shared node cannot also be one of the arms.
    keep = center != other
    center, other = center[keep], other[keep]
    if center.numel() == 0:
        return empty

    # 3. Deduplicate (center, other) so parallel/duplicate edges add no
    #    redundant arms. Sorting the packed key also groups by center (step 4).
    pair_key = torch.stack((center, other), dim=1)
    pair_key = torch.unique(pair_key, dim=0)  # sorted lexicographically by (center, other)
    center, other = pair_key[:, 0].contiguous(), pair_key[:, 1].contiguous()

    # 4. Per-center contiguous groups (already sorted by `unique`).
    uniq_c, counts = torch.unique_consecutive(center, return_counts=True)
    group_start = counts.cumsum(0) - counts                 # first flat index of each group
    deg = counts                                            # neighbours per center

    # Each element pairs with the neighbours that follow it inside its group.
    # For flat element p at local position l in a group of size k:
    #   #partners(p) = k - 1 - l
    pos = torch.arange(center.numel(), device=device)
    local_pos = pos - group_start.repeat_interleave(deg)    # l for every element
    group_deg = deg.repeat_interleave(deg)                  # k for every element
    n_partners = group_deg - 1 - local_pos                  # partners following p

    total_pairs = int(n_partners.sum())
    if total_pairs == 0:
        return empty

    # 5. Vectorised segmented pair generation.
    #    `first`  : repeat each element p, n_partners[p] times.
    #    `second` : p+1, p+2, ... p+n_partners[p]  (the following neighbours).
    first = pos.repeat_interleave(n_partners)
    block_start = (n_partners.cumsum(0) - n_partners)       # start offset in pair space per element
    within = torch.arange(total_pairs, device=device) - block_start.repeat_interleave(n_partners)
    second = first + 1 + within

    a = other[first]                  # one arm
    c = center[first]                 # shared node (== center[second])
    b = other[second]                 # other arm; sorted => a < b, a != b guaranteed

    return torch.stack((a, c, b), dim=0).to(dtype)


def hyperedge_to_incidence(hyperedge_matrix: torch.Tensor,
                           hyperedge_attr: torch.Tensor):
    """Convert an N x M hyperedge matrix into a 2 x Q hyperedge incidence
    matrix and a 1 x R hyperedge attribute tensor, also producing a
    conjugate (node-order-reversed) copy for each hyperedge.

    Args:
        hyperedge_matrix: ``(N, M)`` int64 tensor. Entry ``(i, j)`` is the
            index of the i-th node of the j-th hyperedge.
        hyperedge_attr: ``(1, M)`` tensor of per-hyperedge attributes.

    Returns:
        incidence: ``(2, Q)`` int64 tensor. Row 0 holds node indices, row 1
            holds the corresponding hyperedge indices. ``Q = R * N``.
        new_attr: ``(1, R)`` tensor where ``R = 2 * M``. Hyperedges
            ``[0, M)`` are the originals; ``[M, 2M)`` are their conjugates,
            inheriting the original attributes.
    """
    if hyperedge_matrix.dim() != 2:
        raise ValueError(f"hyperedge_matrix must be 2-D, got shape {tuple(hyperedge_matrix.shape)}")
    N, M = hyperedge_matrix.shape

    attr_flat = hyperedge_attr.reshape(-1)
    if attr_flat.numel() != M:
        raise ValueError(f"hyperedge_attr must have M={M} elements, got {attr_flat.numel()}")

    device = hyperedge_matrix.device
    he = hyperedge_matrix.to(torch.long)

    conjugates = torch.flip(he, dims=[0])              # (N, M)
    all_he = torch.cat([he, conjugates], dim=1)        # (N, 2M)
    R = 2 * M

    node_indices = all_he.t().reshape(-1)              # (2M*N,) row-major over hyperedges
    hyperedge_indices = torch.arange(R, device=device, dtype=torch.long).repeat_interleave(N)

    incidence = torch.stack([node_indices, hyperedge_indices], dim=0)
    new_attr = torch.cat([attr_flat, attr_flat])
    return incidence, new_attr

def get_unique_elements_first_idx(tensor):
    unique, inverse = torch.unique(tensor, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    first_indices = scatter(perm, inverse, -1, reduce="min")

    return first_indices, unique

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
        average="cc",
        dtype="float32"
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
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      nl_ij = neighborlist_soap.compute(points=pos.to(device),
                                        box=float('inf') * torch.eye(3,dtype=torch.float64).to(device),
                                        periodic=False,
                                        quantities="P")[0]
      sorted_values, sorted_indices = torch.sort(nl_ij[:, 0])
      sorted_nl_ij = nl_ij[sorted_indices]
      first_occurence_indices, unique_values = get_unique_elements_first_idx(sorted_values)
      first_occurence_indices = torch.cat((first_occurence_indices,torch.tensor([len(sorted_values)]).to(device)))
      subsystems = []
      subsystem_centers = [[0] for _ in range(len(unique_values))]
      for i in range(len(unique_values)):
          first_index = first_occurence_indices[i]
          final_index = first_occurence_indices[i+1]
          nl_indices = torch.cat([torch.tensor([unique_values[i]]).to(device),sorted_nl_ij[first_index:final_index,1]]).cpu().numpy()
          subsystems.append(system[nl_indices])

      soap_descriptors = torch.from_numpy(soap.create(subsystems, centers=subsystem_centers,n_jobs=-1))
     
      for i in range(len(element_array)):
        one_hot_encoded = torch.zeros(2)
        if number[i] == 1:
            one_hot_encoded = torch.tensor([1, 0])
        if number[i] == 8:
            one_hot_encoded = torch.tensor([0, 1])
        soap_fea.append(torch.hstack((one_hot_encoded, soap_descriptors[i])))
       
      del subsystems, subsystem_centers, soap_descriptors, system, nl_ij, sorted_values, sorted_indices, sorted_nl_ij, nl_indices, first_occurence_indices, unique_values

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
       number_tensor = torch.from_numpy(number).to(device)
       nl_ij, nl_d = neighborlist_chemgnn.compute(points=pos,
                                                  box=float('inf') * torch.eye(3,dtype=torch.float64).to(device),
                                                  periodic=False,
                                                  quantities="Pd")
       nl_atom_i = number_tensor[nl_ij[:,0]]
       nl_atom_j = number_tensor[nl_ij[:,1]]
       nl_atom_pair = list(zip(nl_atom_i.tolist(),nl_atom_j.tolist()))
       nl_cutoffs = torch.tensor(list(map(cutoffs.get, nl_atom_pair))).to(device)
       nl_connected_indices = (nl_d < nl_cutoffs).nonzero(as_tuple=True)[0]
       edge_index = nl_ij[nl_connected_indices,:].t()
       edge_attr = nl_d[nl_connected_indices]

       del number_tensor, nl_ij, nl_d, nl_atom_i, nl_atom_j, nl_atom_pair, nl_cutoffs, nl_connected_indices

    c = int(pos.shape[0])
    batch = []
    for i in range(int(c / len(number))):
        batch += [i] * len(number)
    batch = torch.tensor(batch).to(device)

#convert the edge_attr from the old pair-wise format to the new hyperedge incidence matrix format
#for two-body interactions only, i.e., bonds
    hyperedge_base_index = 0
    hyperedge_bond_index, hyperedge_bond_attr = hyperedge_to_incidence(edge_index, edge_attr)
    hyperedge_bond_index[1,:] = hyperedge_bond_index[1,:] + hyperedge_base_index
#for three-body interactions only, i.e. angles 
    angle_index = edges_to_angles(edge_index)
    angle_attr = compute_angles(pos, angle_index)
    hyperedge_base_index = hyperedge_base_index + hyperedge_bond_attr.size(0)
    hyperedge_angle_index, hyperedge_angle_attr  = hyperedge_to_incidence(angle_index, angle_attr)
    hyperedge_angle_index[1,:] = hyperedge_angle_index[1,:] +  hyperedge_base_index
#concatecate the two-body and three-body interactions 
    hyperedge_index = torch.cat((hyperedge_bond_index,hyperedge_angle_index),dim=1)
    hyperedge_attr = torch.cat((hyperedge_bond_attr,hyperedge_angle_attr))

    x = dict({
        "x": x_full,
        "hyperedge_index": hyperedge_index,
        "hyperedge_attr": hyperedge_attr,
        "batch": batch
    })
    return x 

def generate_soap_energy(number, pos, neighborlist_soap=None):
    tem = []
    element_array = [element_map[num] for num in number]
    element_string = ''.join(element_array)
    soap = SOAP(
        species=set(number),
        r_cut=10.0,
        n_max=10,
        l_max=5,
        periodic=False,
        average="outer",
        dtype="float32"
    )

    # for training and benchmarking when a neighborlist is not applicable
    if (neighborlist_soap is None):
      system = Atoms(symbols=element_string, positions=pos)
      for iatom in range(len(element_array)):
        output_stream = io.StringIO()
        with contextlib.redirect_stdout(output_stream):
            soap_descriptors = torch.from_numpy(soap.create(system, centers=[iatom],n_jobs=-1,verbose=False))
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
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      nl_ij = neighborlist_soap.compute(points=pos.to(device),
                                        box=float('inf') * torch.eye(3,dtype=torch.float64).to(device),
                                        periodic=False,
                                        quantities="P")[0]
      sorted_values, sorted_indices = torch.sort(nl_ij[:, 0])
      sorted_nl_ij = nl_ij[sorted_indices]
      first_occurence_indices, unique_values = get_unique_elements_first_idx(sorted_values)
      first_occurence_indices = torch.cat((first_occurence_indices,torch.tensor([len(sorted_values)]).to(device)))
      subsystems = []
      subsystem_centers = [[0] for _ in range(len(unique_values))]
      for i in range(len(unique_values)):
          first_index = first_occurence_indices[i]
          final_index = first_occurence_indices[i+1]
          nl_indices = torch.cat([torch.tensor([unique_values[i]]).to(device),sorted_nl_ij[first_index:final_index,1]]).cpu().numpy()
          subsystems.append(system[nl_indices])

      output_stream = io.StringIO()
      with contextlib.redirect_stdout(output_stream):
          soap_descriptors = torch.from_numpy(soap.create(subsystems, centers=subsystem_centers,n_jobs=1,verbose=False))
     
      for iatom in range(len(element_array)):                           
         one_hot_encoded = torch.zeros(2)
         if number[iatom] == 1:
             one_hot_encoded = torch.tensor([1, 0])
         if number[iatom] == 8:
             one_hot_encoded = torch.tensor([0, 1])
         tem.append(torch.hstack((one_hot_encoded, soap_descriptors[iatom])))

      del subsystems, subsystem_centers, soap_descriptors, output_stream, system, nl_ij, sorted_values, sorted_indices, sorted_nl_ij, nl_indices, first_occurence_indices, unique_values

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
       number_tensor = torch.from_numpy(number).to(device)
       nl_ij, nl_d = neighborlist_chemgnn.compute(points=pos,
                                                  box=float('inf') * torch.eye(3,dtype=torch.float64).to(device),
                                                  periodic=False,
                                                  quantities="Pd")
       nl_atom_i = number_tensor[nl_ij[:,0]]
       nl_atom_j = number_tensor[nl_ij[:,1]]
       nl_atom_pair = list(zip(nl_atom_i.tolist(),nl_atom_j.tolist()))
       nl_cutoffs = torch.tensor(list(map(cutoffs.get, nl_atom_pair))).to(device)
       nl_connected_indices = (nl_d < nl_cutoffs).nonzero(as_tuple=True)[0]
       edge_index = nl_ij[nl_connected_indices,:].t()
       edge_attr = nl_d[nl_connected_indices]

       del number_tensor, nl_ij, nl_d, nl_atom_i, nl_atom_j, nl_atom_pair, nl_cutoffs, nl_connected_indices

    c = int(pos.shape[0])
    batch = []
    for i in range(int(c / len(number))):
        batch += [i] * len(number)
    batch = torch.tensor(batch).to(device)

#convert the edge_attr from the old pair-wise format to the new hyperedge incidence matrix format
#for two-body interactions only, i.e., bonds
    hyperedge_base_index = 0
    hyperedge_bond_index, hyperedge_bond_attr = hyperedge_to_incidence(edge_index, edge_attr)
    hyperedge_bond_index[1,:] = hyperedge_bond_index[1,:] + hyperedge_base_index
#for three-body interactions only, i.e. angles 
    angle_index = edges_to_angles(edge_index)
    angle_attr = compute_angles(pos, angle_index)
    hyperedge_base_index = hyperedge_base_index + hyperedge_bond_attr.size(0)
    hyperedge_angle_index, hyperedge_angle_attr  = hyperedge_to_incidence(angle_index, angle_attr)
    hyperedge_angle_index[1,:] = hyperedge_angle_index[1,:] +  hyperedge_base_index
#concatecate the two-body and three-body interactions 
    hyperedge_index = torch.cat((hyperedge_bond_index,hyperedge_angle_index),dim=1)
    hyperedge_attr = torch.cat((hyperedge_bond_attr,hyperedge_angle_attr))

    x = dict({
        "x": x_full,
        "hyperedge_index": hyperedge_index,
        "hyperedge_attr": hyperedge_attr,
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

