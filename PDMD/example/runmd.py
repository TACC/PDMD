import sys
import os

#Get the absolute root path to the PDMD package
pdmd_module_path=os.path.join(os.path.dirname(__file__),'../../')

#Append the PDMD root path to sys.path
sys.path.append(pdmd_module_path)

import torch
import ase
from ase.io import read
from ase.io import write
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nose_hoover_chain import NoseHooverChainNVT
import PDMD
from PDMD.benchmark.ChemGNN_calculator import ChemGNN_Calculator

#set the path to the model files, i.e., energy.pt and forces.pt
model_path = os.path.dirname(os.path.abspath(PDMD.__file__))+"/benchmark"

#set the filename to the input XYZ file for the initial geometry of a water cluster
input_xyz_filename = str(sys.argv[1])

#read a water cluster from the designated input XYZ file
water = read(input_xyz_filename)

#define an ChemGNN calculator 
chemgnn_calculator = ChemGNN_Calculator(model_path+"/energy.pt",model_path+"/forces.pt","md")

#attach the ChemGNN calculator to the water cluster
water.set_calculator(chemgnn_calculator)

#set the initial temperature to 300.0K
temperature = 300.0
MaxwellBoltzmannDistribution(water,temperature_K=temperature)

#set up an NVT molecular dynamics simulation using the NoseHooverChain thermostat
timestep = 0.25*units.fs
tdamp = 10.0*timestep
md = NoseHooverChainNVT(water, timestep=timestep, temperature_K=temperature, tchain=10, tloop=10, tdamp=tdamp)

#run an MD simulation with nsteps
nsteps = 10000
for istep in range(nsteps):
 #get the water cluster's potential energy
 energy = str(water.get_potential_energy()).lstrip('[').rstrip(']')
 #get the water cluster's temperature
 md_temperature = water.get_temperature()
 #print out some MD information
 print(f"Step: {istep:6d} Temperature: {md_temperature:6.2f}K {energy:10s} eV")

 #run one MD step
 md.run(1)

#delete the water cluster's calculator
water.calc = None
