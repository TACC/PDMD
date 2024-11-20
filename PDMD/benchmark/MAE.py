import numpy as np

def read_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            numbers = line.strip().split()
            data.append([float(number) for number in numbers])
    return np.array(data)

def calculate_MAE():
    average = []
    for i in range(1, 22):
        # load data
        data_DFT_energy = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/DFT_ENERGY_WAT{i}_1120')
        data_ML_energy = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/ML_ENERGY_WAT{i}_1120')
        assert data_DFT_energy.shape == data_ML_energy.shape, "Data shapes are not identical!"
        # calculate MAE
        mae_energy = np.mean(np.abs(data_DFT_energy - data_ML_energy))
        # unit conversion
        mae_energy = mae_energy * 27.211 * 1000 / (i * 3)
        print(f"MAE_ENERGY WATER{i}: {mae_energy}")

        data_DFT_forces = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/DFT_FORCES_WAT{i}_1120')
        data_ML_forces = read_data(f'./PDMD/benchmark/BENCHMARK_ML_4/ML_FORCES_WAT{i}_1120')
        assert data_DFT_forces.shape == data_ML_forces.shape, "Data shapes are not identical!"
        # calculate MAE
        mae_forces = np.mean(np.abs(data_DFT_forces - data_ML_forces))
        # unit conversion
        mae_forces = mae_forces * 27.211 * 1000 / 0.529177
        average.append(mae_forces)
        print(f"MAE_FORCES WATER{i}: {mae_forces}")
    average = np.asarray(average)
    average = np.mean(average)
    print(average)

if __name__ == '__main__':
    calculate_MAE()
