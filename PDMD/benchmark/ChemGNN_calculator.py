from ase.calculators.calculator import Calculator
import ase
import torch
from PDMD.benchmark.ChemGNN_energy import ChemGNN_EnergyModel
from PDMD.benchmark.ChemGNN_forces import ChemGNN_ForcesModel

#define a customarized ASE Calculator
class ChemGNN_Calculator(Calculator):

 implemented_properties = ['energy','forces'] 
 
 def __init__(self,energy_pth_filename,forces_pth_filename):
  Calculator.__init__(self)
  self.energy_pth_filename = energy_pth_filename
  self.forces_pth_filename = forces_pth_filename
  #filename for the energy model in .pth format
  #print("Energy PTH File: ",self.energy_pth_filename)
  #filename for the force model in .pth format
  #force_pth_filename
  #print("Force PTH File: ",self.forces_pth_filename)

  #load energy .pt model
  self.energy_model = ChemGNN_EnergyModel()
  self.energy_model = self.energy_model.to(dtype=torch.float32)
  checkpoint_energy = torch.load(self.energy_pth_filename,
                                 map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 weights_only=False)
  energy_model_state_dict = checkpoint_energy["model_state_dict"]
  self.energy_model.load_state_dict(energy_model_state_dict)

  # load forces .pt model
  self.forces_model = ChemGNN_ForcesModel()
  self.forces_model = self.forces_model.to(dtype=torch.float32)
  checkpoint_forces = torch.load(self.forces_pth_filename,
                                 map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 weights_only=False)
  forces_model_state_dict = checkpoint_forces["model_state_dict"]
  self.forces_model.load_state_dict(forces_model_state_dict)


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


  #convert positions to a torch tensor
  tensor_positions = torch.tensor(positions)
  energy = self.energy_model(atomic_numbers,tensor_positions)
  forces = self.forces_model(atomic_numbers,tensor_positions)


  #energy and forces unit conversion
  #the units for energy and forces in machine learning are Hartree and Hartree/Bohr, respectively
  #while the untis for ASE are eV and eV/Angstrom, respectively
  # energy = energy*units.Hartree
  # forces = forces*units.Hartree/units.Bohr

  # print("Energy: ",energy)
  #print("Forces: ",forces)
  # print("MAD:", mad)
  self.results = {
       "energy": energy,
       "forces": forces,
  }

 # emadfile = open('BENCHMARK'+"/ML_ENERGY_MAD_WAT"+str(cluster_size)+"_1000","a")
 # emadfile.write()
 # emadfile.write("\n")
