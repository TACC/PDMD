import torch
from ase.io import read
from PDMD.benchmark.ChemGNN_calculator import ChemGNN_Calculator

#set directory of benchmark data sets
# benchmark_dataset = {'BENCHMARK_DFT':1000}
benchmark_dataset = {'./PDMD/benchmark/BENCHMARK_ML_4':1120}

def benchmark():
 for benchmark_dir, benchmark_size in benchmark_dataset.items():
 #set water cluster size
  cluster_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
  for cluster_size in cluster_sizes:
   #read a water molecule from the water.xyz file
   water_molecules = read(filename=benchmark_dir+'/water_'+str(cluster_size)+'_'+str(benchmark_size)+'.xyz',index=':')
   #retrieve the number of molecules
   nmolecules = len(water_molecules)
   print("Number of Molecules: ",nmolecules)
   #retrieve the number of atoms
   natoms = len(water_molecules[0])
   print("Number of Atoms: ",natoms)
   #retrieve the atomic numbers
   #atomic_numbers = water_molecules[0].get_atomic_numbers()
   #print("Atomic Numbers: ",atomic_numbers)
   #retrieve the positions
   #positions = water_molecules[0].get_positions()
   #print("Positions: ",positions)

   #set water system to the first snapshot
   water = water_molecules[0]
   #set the system's calculator to ChemGNN
   water.calc=ChemGNN_Calculator("./PDMD/benchmark/energy.pt", "./PDMD/benchmark/forces.pt")
   #set the output precision
   #set the energy and forces output filenames
   efile = open(benchmark_dir+"/ML_ENERGY_WAT"+str(cluster_size)+"_"+str(benchmark_size),"a")
   #emadfile = open(benchmark_dir+"/ML_ENERGY_MAD_WAT"+str(cluster_size)+"_"+str(benchmark_size),"a")
   ffile = open(benchmark_dir+"/ML_FORCES_WAT"+str(cluster_size)+"_"+str(benchmark_size),"a")
   #fmadfile = open(benchmark_dir + "/ML_FORCE_MAD_WAT" + str(cluster_size) + "_"+str(benchmark_size), "a")

   #start validating the energy and forces for each loaded strcutures
   for imol in range(nmolecules):
    print("Validating Structure_",imol+1)
    #set the system's position
    water.set_positions(water_molecules[imol].get_positions())
    #output the system's energy
    energy = water.get_potential_energy()
    # force = water.get_forces()
    efile.write(str(energy.item()))
    efile.write("\n")
    #efile.flush()

    #energy_mad = water.calc.results["energy_mad"]
    #emadfile.write(str(energy_mad))
    #emadfile.write("\n")
    #output the system's forces
    forces = water.get_forces()
    ffile.write(str(forces.detach().numpy()).replace('[','').replace(']',''))
    ffile.write("\n")

    #forces_mad = water.calc.results["forces_mad"]
    #fmadfile.write(str(forces_mad))
    #fmadfile.write("\n")
    #ffile.flush()
   efile.close()
   #emadfile.close()
   ffile.close()
   #fmadfile.close()

   #delete the system's calculator
   del water.calc

