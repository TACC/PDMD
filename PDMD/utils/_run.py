import time
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import os.path as osp
import os
import numpy as np
import torch
import random
import json
from ..models import ENERGY_Model, FORCE_Model
from . import get_timestring, MutilWaterDataset, split_dataset, worker_init_fn, train, val, draw_two_dimension, reverse_min_max_scaler_1d


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# torch.set_default_dtype(torch.float64)

def run(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    config.timestring = get_timestring()
    model_config_print = config.__dict__.copy()
    print("Load full config successfully:")
    print("config =", json.dumps(model_config_print, indent=4))
    config.device = torch.device("cuda" if torch.cuda.is_available() and config.device_type.lower() == "gpu" else "cpu")

    if not os.path.exists(config.process_dst_path):
        answer = input(f"Processed data not found in {config.process_dst_path}. You need to start processing dataset first.")
    else:
        print(f"Processed data is found in {config.process_dst_path}. Continue ...")

    print("[Step 1] Configurations")
    print("using: {}".format(config.device))
    for item in config.__dict__.items():
        if item[0][0] == "_":
            continue
        print("{}: {}".format(item[0], item[1]))

    print("[Step 2] Preparing dataset...")
    dataset_path = osp.join(config.main_path, config.process_dst_path)

    trainset_list = []
    valset_list = []
    assert config.dataset in ["energy_dataset", "force_dataset"]
    data_type = None
    if config.dataset == 'energy_dataset':
        data_type = 'energy'
    elif config.dataset == 'force_dataset':
        data_type = 'force'
    for i in range(1, 22):
        dataset = MutilWaterDataset(root=dataset_path, split=f"{i}water_{data_type}")
        tr_data, v_data, te_data = split_dataset(dataset, train_p=config.train_ratio, val_p=config.val_ratio, shuffle=True)
        trainset_list.append(tr_data)
        valset_list.append(v_data)
    dataset = MutilWaterDataset(root=dataset_path, split=f"water_{data_type}_optimized")
    tr_data, v_data, te_data = split_dataset(dataset, train_p=config.train_ratio, val_p=config.val_ratio, shuffle=True)
    trainset_list.append(tr_data)
    valset_list.append(v_data)

    train_dataset = ConcatDataset(trainset_list)
    val_dataset = ConcatDataset(valset_list)

    g = torch.Generator()
    g.manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)

    print("[Step 3] Initializing model")
    main_save_path = osp.join(config.main_path, "saves", config.timestring)
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = osp.join(main_save_path, "model_last.pt")
    model_restart_path = osp.join(main_save_path, "model_restart.pt")
    initial_model_state_path = osp.join(main_save_path, "initial_state_dict.pth")
    final_model_state_path = osp.join(main_save_path, "final_state_dict.pth")
    figure_save_path_lr = osp.join(main_save_path, "lr.png")
    figure_save_path_gradients = osp.join(main_save_path, "gradients.png")
    figure_save_path_train_loss_whole = osp.join(main_save_path, "loss_train.png")
    figure_save_path_train_loss_last_half = osp.join(main_save_path, "loss_train_last_half.png")
    figure_save_path_train_loss_last_quarter = osp.join(main_save_path, "loss_train_last_quarter.png")
    figure_save_path_val_loss = osp.join(main_save_path, "val_loss.png")
    figure_save_path_combined = osp.join(main_save_path, "loss_train_and_val.png")

    regression_result_train_true = osp.join(main_save_path, "train_true.npy")
    regression_result_train_pred = osp.join(main_save_path, "train_pred.npy")
    regression_result_val_true = f"{main_save_path}/val_true.npy"
    regression_result_val_pred = f"{main_save_path}/val_pred.npy"
    regression_result_test_true = f"{main_save_path}/test_true.npy"
    regression_result_test_pred = f"{main_save_path}/test_pred.npy"

    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    print("figure_save_path_loss_train_whole: {}".format(figure_save_path_train_loss_whole))
    print("figure_save_path_loss_train_last_half: {}".format(figure_save_path_train_loss_last_half))
    print("figure_save_path_loss_train_last_quarter: {}".format(figure_save_path_train_loss_last_quarter))
    print("figure_save_path_loss_val_whole: {}".format(figure_save_path_val_loss))
    print("regression_result_train_true: {}".format(regression_result_train_true))
    print("regression_result_train_pred: {}".format(regression_result_train_pred))
    print("regression_result_val_true: {}".format(regression_result_val_true))
    print("regression_result_val_pred: {}".format(regression_result_val_pred))
    print("regression_result_test_true: {}".format(regression_result_test_true))
    print("regression_result_test_pred: {}".format(regression_result_test_pred))

    assert config.model in ["ChemGNN_energy", "ChemGNN_force"]
    model = None
    if config.model == 'ChemGNN_energy':
        model = ENERGY_Model().to(config.device)
    elif config.model == 'ChemGNN_force':
        model = FORCE_Model().to(config.device)

    if os.path.exists(model_restart_path):
        # a restart file was found
        # load the state dictionary from the restart file
        checkpoint = torch.load(model_restart_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # no restart file was found
        initial_state_dict = model.state_dict()
        torch.save(initial_state_dict, initial_model_state_path)

    if config.model == 'ChemGNN_energy':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.000001)
    elif config.model == 'ChemGNN_force':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=50, min_lr=0.0000001)
    print(model)

    print("[Step 4] Training...")

    log_dir = osp.join(config.main_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = osp.join(log_dir, "record.txt")
    with open(log_file_path, "w") as f:
        f.write("")

    start_time = time.time()
    start_time_0 = start_time

    epoch_train_loss_list = []
    epoch_val_loss_list = []
    lr_list = []
    gradients_list = []

    for epoch in range(1, config.epoch + 1):
        model, train_loss, gradients = train(model, config, train_loader, optimizer)
        val_loss = val(model, config, val_loader)
        scheduler.step(val_loss)
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(val_loss)
        lr_list.append(optimizer.param_groups[0]["lr"])
        gradients_list.append(gradients)

        if epoch % config.epoch_step == 0:
            now_time = time.time()
            test_log = "Epoch [{0:05d}/{1:05d}] Loss_train:{2:.6f} Loss_val:{3:.6f} Gradients:{4:.6f} Lr:{5:.6f} (Time:{6:.6f}s Time total:{7:.2f}min Time remain: {8:.2f}min)".format(
                epoch, config.epoch, train_loss, val_loss, gradients, optimizer.param_groups[0]["lr"],
                now_time - start_time, (now_time - start_time_0) / 60.0,
                (now_time - start_time_0) / 60.0 / epoch * (config.epoch - epoch))
            print(test_log)
            with open(log_file_path, "a") as f:
                f.write(test_log + "\n")
            start_time = now_time
            torch.save(
                {
                    "config": config,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": train_loss
                }, model_save_path)

    # Draw loss
    print("[Step 5] Drawing training result...")
    loss_length = len(epoch_train_loss_list)
    loss_x = range(1, config.epoch + 1)

    # draw gradients
    draw_two_dimension(
        y_lists=[gradients_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(gradients_list[-1], min(gradients_list))],
        line_style_list=["solid"],
        fig_title="Gradients",
        fig_x_label="Epoch",
        fig_y_label="Gradients",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_gradients
    )

    # draw learning rate
    draw_two_dimension(
        y_lists=[lr_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(lr_list[-1], min(lr_list))],
        line_style_list=["solid"],
        fig_title="Learning rate",
        fig_x_label="Epoch",
        fig_y_label="Lr",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_lr
    )

    # draw train and validation loss
    draw_two_dimension(
        y_lists=[epoch_train_loss_list, epoch_val_loss_list],
        x_list=loss_x,
        color_list=["blue", "red"],  # You can specify colors for each curve
        legend_list=["Train Loss", "Validation Loss"],  # Add legends for each curve
        line_style_list=["solid", "dashed"],  # You can specify line styles
        fig_title="Train and Validation loss",
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_combined  # Save the combined plot
    )

    # draw train loss_whole
    draw_two_dimension(
        y_lists=[epoch_train_loss_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_train_loss_list[-1], min(epoch_train_loss_list))],
        line_style_list=["solid"],
        fig_title="Train loss - whole",
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_train_loss_whole
    )

    # draw train loss_last_half
    draw_two_dimension(
        y_lists=[epoch_train_loss_list[-(loss_length // 2):]],
        x_list=loss_x[-(loss_length // 2):],
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_train_loss_list[-1], min(epoch_train_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - last half ({} - Loss{})".format(config.dataset, config.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_train_loss_last_half
    )

    print("[Step 6] Saving final model...")

    final_state_dict = model.state_dict()
    torch.save(final_state_dict, final_model_state_path)
