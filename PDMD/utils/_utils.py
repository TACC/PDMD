import torch
import numpy as np
import os.path as osp
import os
from tqdm import tqdm
from torch_geometric.utils import degree
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances

def get_timestring():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def split_dataset(dataset, train_p=60, val_p=20, shuffle=True):
   """
   dataset: dataset to split
   train_p: training set percentage
   val_p: validating set percentage

   """

   if shuffle:
      ############## Change this if not to seek reproducibility ####
      #random.seed(218632)
      idx = list(range(dataset.len()))
      random.shuffle(idx)
      idx = np.array(idx)
   else:
      idx = np.array(range(dataset.len()))

   len_train = int(len(idx) * train_p)
   len_val = int(len(idx) * val_p)

   train_dataset = dataset[idx[0:len_train]]
   val_dataset = dataset[idx[len_train:len_train+len_val]]
   test_dataset = dataset[idx[len_train+len_val:]]
   return train_dataset, val_dataset, test_dataset


def draw_two_dimension(
        y_lists,
        x_list,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6)
) -> None:
   """
   Draw a 2D plot of several lines
   :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
   :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
   :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
   :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
   :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
   :param legend_fontsize: (float) legend fontsize. e.g., 15
   :param fig_title: (string) title of the figure. e.g., "Anonymous"
   :param fig_x_label: (string) x label of the figure. e.g., "time"
   :param fig_y_label: (string) y label of the figure. e.g., "val"
   :param show_flag: (boolean) whether you want to show the figure. e.g., True
   :param save_flag: (boolean) whether you want to save the figure. e.g., False
   :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
   :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
   :param fig_title_size: (float) figure title size. e.g., 20
   :param fig_grid: (boolean) whether you want to display the grid. e.g., True
   :param marker_size: (float) marker size. e.g., 0
   :param line_width: (float) line width. e.g., 1
   :param x_label_size: (float) x label size. e.g., 15
   :param y_label_size: (float) y label size. e.g., 15
   :param number_label_size: (float) number label size. e.g., 15
   :param fig_size: (tuple) figure size. e.g., (8, 6)
   :return:
   """
   assert len(y_lists[0]) == len(x_list), "Dimension of y should be same to that of x"
   assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
   y_count = len(y_lists)
   plt.figure(figsize=fig_size)
   for i in range(y_count):
      plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i],
               linestyle=line_style_list[i])
   plt.xlabel(fig_x_label, fontsize=x_label_size)
   plt.ylabel(fig_y_label, fontsize=y_label_size)
   plt.tick_params(labelsize=number_label_size)
   if legend_list:
      plt.legend(legend_list, fontsize=legend_fontsize)
   if fig_title:
      plt.title(fig_title, fontsize=fig_title_size)
   if fig_grid:
      plt.grid(True)
   if save_flag:
      plt.savefig(save_path, dpi=save_dpi)
   if show_flag:
      plt.show()
   plt.clf()
   plt.close()


def reverse_min_max_scaler_1d(data_normalized, data_min=-361.77515914, data_max=-17.19880547, new_min=0.0,
                              new_max=1.0):
   core = (data_normalized - new_min) / (new_max - new_min)
   data_original = core * (data_max - data_min) + data_min
   return data_original


def worker_init_fn(worker_id, seed=0):
   random.seed(seed + worker_id)