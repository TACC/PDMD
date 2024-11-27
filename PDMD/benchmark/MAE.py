import numpy as np

def read_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            numbers = line.strip().split()
            data.append([float(number) for number in numbers])
    return np.array(data)

def calculate_MAE():
    average_forces = []
    average_energy = []
    for i in range(1, 22):
        # load data
        data_DFT_energy = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/DFT_ENERGY_WAT{i}_1120')
        data_ML_energy = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/ML_ENERGY_WAT{i}_1120')
        assert data_DFT_energy.shape == data_ML_energy.shape, "Data shapes are not identical!"
        # calculate MAE
        mae_energy = np.mean(np.abs(data_DFT_energy - data_ML_energy))
        # unit conversion
        mae_energy = mae_energy * 27.211 * 1000 / (i * 3)
        print("MAE_ENERGY WATER%r: %6.2f meV/angstrom/atom" %(i, mae_energy))

        data_DFT_forces = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/DFT_FORCES_WAT{i}_1120')
        data_ML_forces = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/ML_FORCES_WAT{i}_1120')
        assert data_DFT_forces.shape == data_ML_forces.shape, "Data shapes are not identical!"
        # calculate MAE
        mae_forces = np.mean(np.abs(data_DFT_forces - data_ML_forces))
        # unit conversion
        mae_forces = mae_forces * 27.211 * 1000 / 0.529177
        average_forces.append(mae_forces)
        average_energy.append(mae_energy)
        print("MAE_FORCES WATER%r: %6.2f meV/angstrom" %(i,mae_forces))
    average_forces = np.asarray(average_forces)
    average_forces = np.mean(average_forces)
    average_energy = np.asarray(average_energy)
    average_energy = np.mean(average_energy)
    print("")
    print("AVERAGE MAE_ENERGY:%7.2f meV/angstrom" %(average_energy))
    print("AVERAGE MAE_FORCES:%7.2f meV/angstrom/atom" %(average_forces))
    print("")

if __name__ == '__main__':
    calculate_MAE()
