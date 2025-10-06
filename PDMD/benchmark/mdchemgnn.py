import torch
from ase.io import read
from PDMD.benchmark.ChemGNN_calculator import ChemGNN_Calculator
import numpy as np
import sys

#set directory of benchmark data sets
benchmark_dir = './PDMD/benchmark/BENCHMARK_DATASET'
#set benchemark_size:cluster_sizes
benchmark_dataset = {
        1120:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,40,50,60,70,80,90,100],
        112:[200,300,400,500,600,700,800,900,1000],
}

def benchmark():
 #set the output precision
 np.set_printoptions(threshold=sys.maxsize)
 for benchmark_size, cluster_sizes in benchmark_dataset.items():
  for cluster_size in cluster_sizes:
   #read a water molecule from the water.xyz file
   water_molecules = read(filename=benchmark_dir+'/water_'+str(cluster_size)+'_'+str(benchmark_size)+'.xyz',index=':')
   #retrieve the number of molecules
   nmolecules = len(water_molecules)
   print("Number of Molecules: ",nmolecules)
   #retrieve the number of atoms
   natoms = len(water_molecules[0])
   print("Number of Atoms: ",natoms)

   #set water system to the first snapshot
   water = water_molecules[0]
   #set the system's calculator to ChemGNN
   water.calc=ChemGNN_Calculator("./PDMD/benchmark/energy.pt", "./PDMD/benchmark/forces.pt")
   #set the energy and forces output filenames
   efile = open(benchmark_dir+"/ML_ENERGY_WAT"+str(cluster_size)+"_"+str(benchmark_size),"a")
   ffile = open(benchmark_dir+"/ML_FORCES_WAT"+str(cluster_size)+"_"+str(benchmark_size),"a")

   #start validating the energy and forces for each loaded strcutures
   for imol in range(nmolecules):
    print("Validating Structure_",imol+1)
    #set the system's position
    water.set_positions(water_molecules[imol].get_positions())
    #output the system's energy
    energy = water.get_potential_energy()
    efile.write(str(energy.item()))
    efile.write("\n")
    #efile.flush()
    #output the system's forces
    forces = water.get_forces()
    ffile.write(str(forces.detach().numpy()).replace('[','').replace(']',''))
    ffile.write("\n")
    #ffile.flush()

   efile.close()
   ffile.close()

   #delete the system's calculator
   del water.calc

