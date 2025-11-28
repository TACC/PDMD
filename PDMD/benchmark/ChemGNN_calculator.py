from ase.calculators.calculator import Calculator
import ase
from ase import units
from ase.neighborlist import NewPrimitiveNeighborList
import torch
from PDMD.benchmark.ChemGNN_energy import ChemGNN_EnergyModel
from PDMD.benchmark.ChemGNN_forces import ChemGNN_ForcesModel
import numpy as np

#define a customarized ASE Calculator
class ChemGNN_Calculator(Calculator):

 implemented_properties = ['energy','forces'] 
 
 def __init__(self,energy_pth_filename,forces_pth_filename,runtype):
  Calculator.__init__(self)
  self.energy_pth_filename = energy_pth_filename
  self.forces_pth_filename = forces_pth_filename
  #filename for the energy model in .pth format
  #print("Energy PTH File: ",self.energy_pth_filename)
  #filename for the force model in .pth format
  #force_pth_filename
  #print("Force PTH File: ",self.forces_pth_filename)
  self.runtype = runtype
  #runtype: benchmark or md for unit conversion
  #print("Run Type: ",self.runtype)
  # for MD runs, construct a neighbor list
  if (self.runtype == "md"):
   #set up a neighor list for the system
   #cutoff distance set to 10.0 angstrom
   nl_cutoffs = 10.0
   #buffer thickness set to 1.0 angstrom
   nl_skin = 1.0
   #no need to sort the neighbor list
   nl_sorted = False
   #include an atom into its neighborlist
   nl_self = True
   #double-count the neighborlist
   nl_bothways = True
   #do not use scaled positions
   nl_use_scaled_positions = False
   self.neighborlist = NewPrimitiveNeighborList(cutoffs = nl_cutoffs, skin = nl_skin, sorted = nl_sorted, self_interaction = nl_self, bothways = nl_bothways, use_scaled_positions = nl_use_scaled_positions)
   print("NEIGHBOR LIST CONSTRUCTED")

  # allocation CMA for connectivity cutoff matrix
  print("SETTING CMA TO 0")
  CMA = torch.empty((0,0))

  #load energy .pt model
  self.energy_model = ChemGNN_EnergyModel()
  self.energy_model = self.energy_model.to(dtype=torch.float32)
  checkpoint_energy = torch.load(self.energy_pth_filename,
                                 map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 weights_only=False)

  energy_model_state_dict = checkpoint_energy["model_state_dict"]
  energy_feature_min_values = checkpoint_energy["min_energy_values"]
  energy_feature_max_values = checkpoint_energy["max_energy_values"]
  self.energy_model.load_state_dict(energy_model_state_dict)
  self.energy_model.CMA = CMA
  self.energy_feature_min_values = energy_feature_min_values
  self.energy_feature_max_values = energy_feature_max_values

  # load forces .pt model
  self.forces_model = ChemGNN_ForcesModel()
  self.forces_model = self.forces_model.to(dtype=torch.float32)
  checkpoint_forces = torch.load(self.forces_pth_filename,
                                 map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 weights_only=False)

  forces_model_state_dict = checkpoint_forces["model_state_dict"]
  forces_feature_min_values = checkpoint_forces["min_force_values"]
  forces_feature_max_values = checkpoint_forces["max_force_values"]
  self.forces_model.load_state_dict(forces_model_state_dict)
  self.forces_model.CMA = CMA
  self.forces_feature_min_values = forces_feature_min_values
  self.forces_feature_max_values = forces_feature_max_values

 def __del__(self):
  #delete energy model
  #print("Deleteing energy mode")
  del self.energy_model
  #delete forces model
  #print("Deleteing forces mode")
  del self.forces_model
   
 def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=ase.calculators.calculator.all_changes):
  ase.calculators.calculator.Calculator.calculate(self,atoms,properties,system_changes) 
  #print("Number of atoms: ",len(atoms))
  atomic_numbers = atoms.get_atomic_numbers()
  #print("Atomic Numbers: ",atomic_numbers)
  positions = atoms.get_positions()
  #print("Positions: ",positions)
  #place holders for energy and forces using random numbers
  self.energy_model.eval()
  self.forces_model.eval()

  if (self.runtype == "md"):
   self.neighborlist.update(atoms.pbc, atoms.get_cell(), atoms.positions)
   #print("N_UPDATES: ",self.neighborlist.nupdates)

  #convert positions to a torch tensor
  tensor_positions = torch.tensor(positions)
  energy = self.energy_model(atomic_numbers,tensor_positions, self.energy_feature_min_values, self.energy_feature_max_values, self.neighborlist)
  forces = self.forces_model(atomic_numbers,tensor_positions, self.forces_feature_min_values, self.forces_feature_max_values, self.neighborlist)

  #energy and forces unit conversion
  #the units for energy and forces in machine learning are Hartree and Hartree/Bohr, respectively
  #while the untis for ASE are eV and eV/Angstrom, respectively
  if (self.runtype == "md"):
   energy = energy*units.Hartree
   forces = forces*units.Hartree/units.Bohr
   forces = forces.detach().numpy()

  self.results = {
       "energy": energy,
       "forces": forces,
  }

