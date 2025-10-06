import time
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import os.path as osp
import os
import warnings
import numpy as np
import torch
import lightning
import random
import json
from ..models import ENERGY_Model, FORCE_Model
from . import get_timestring, MutilWaterDataset, split_dataset, worker_init_fn, train, val, draw_two_dimension, reverse_min_max_scaler_1d

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# define a LightningModule named ChemLightning 
class ChemLightning(lightning.LightningModule):
    # initialiation of a ChemLightning object: 
    # chemmodel, config, factor, patience and min_lr can be accessed as hyperparameters
    # for example, self.hparams.factor
    def __init__(self, chemmodel, config, model_save_path, factor=0.6, patience=30, min_lr=1.0e-6):
        super().__init__()
        #initiate the Chemistry model
        self.chemmodel = chemmodel 
        #initiate the model's hyperparameters 
        if config.model == 'ChemGNN_energy':
         factor=0.6
         patience=30
         min_lr=1.0e-6
        if config.model == 'ChemGNN_forces':
         factor=0.6
         patience=50
         min_lr=1.0e-7
        #save the model's hyperparameters
        self.save_hyperparameters(ignore=['chemmodel'])
        #disable automatic optimization by LightingModule, alternatively, use your own optimizer and scheduler
        self.automatic_optimization = False

    # initiate the optimizer and scheduler, called only once during class object initiazation
    def configure_optimizers(self):
        #setup optimizer
        if self.hparams.config.model == 'ChemGNN_energy':
         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.config.lr)
        if self.hparams.config.model == 'ChemGNN_forces':
         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config.lr)
        #setup scheduler 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.factor, patience=self.hparams.patience, min_lr=self.hparams.min_lr)
        return {"optimizer":optimizer,"lr_scheduler":scheduler}

    # for each step of model training
    # note that only one record is avaiable in the input parameter "batch", i.e., batch_size = 1
    # no need to iterate over train_loader
    def training_step(self, batch, batch_idx):
        model = self.chemmodel
        config = self.hparams.config
        optimizer = self.optimizers().optimizer
        train_loader = self.trainer.train_dataloader
        model_name = model.model_name
        assert model_name in ["ChemGNN_energy", "ChemGNN_forces"]
        total_loss = 0.0
        gradients_list = []
        if model_name == "ChemGNN_energy":
            optimizer.zero_grad()
            input_dict = dict({
                "x": batch.x,
               "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "batch": batch.batch
            })
            out = model(input_dict)

            mybatch = input_dict["batch"]
            node_counts = torch.bincount(mybatch)

            train_loss = (((out.squeeze() - batch.y) / node_counts).abs()).mean()
            train_loss.backward()
            total_loss += train_loss.item() * batch.num_graphs
            total_count = batch.num_graphs
            
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and name == 'energy_predictor.4.weight':
                    gradients = torch.norm(parameter.grad, p=2)
                    gradients = gradients.item()
                    gradients_list.append(gradients)
                    break
            optimizer.step()
        if model_name == "ChemGNN_forces": 
            optimizer.zero_grad()
            input_dict = dict({
                "x": batch.x,
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "batch": batch.batch
            })
            out = model(input_dict)

            train_loss = (out.squeeze() - batch.z).abs().mean()
            train_loss.backward()
            total_loss = train_loss.item() * batch.num_nodes
            total_count = batch.num_nodes

            for name, parameter in model.named_parameters():
                if parameter.requires_grad and name == 'force_predictor.2.weight':
                    gradients = torch.norm(parameter.grad, p=2)
                    gradients = gradients.item()
                    gradients_list.append(gradients)
                    break
            optimizer.step()

        self.log("train_loss", total_loss, batch_size=config.batch_size, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_count", total_count, batch_size=config.batch_size, on_step=False, on_epoch=True, sync_dist=True)
        return train_loss

    # for each step of model validation
    # note that only one record is avaiable in the input parameter "batch", i.e., batch_size = 1
    # no need to iterate over val_loader
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
       model = self.chemmodel
       config = self.hparams.config
       val_loader = self.trainer.val_dataloaders
       # need to rewrite using your val() function as template
       # val_loss = val(model, config, val_loader)
       model_name = model.model_name
       assert model_name in ["ChemGNN_energy", "ChemGNN_forces"]
       if model_name == "ChemGNN_energy":
               input_dict = dict({
                "x": batch.x,
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "batch": batch.batch
               })
               out = model(input_dict)
               mybatch = input_dict["batch"]
               node_counts = torch.bincount(mybatch)

               val_loss = (((out.squeeze() - batch.y) / node_counts).abs()).mean()
               total_error = val_loss.item() * batch.num_graphs
               total_count = batch.num_graphs
           
       if model_name == "ChemGNN_forces":
               input_dict = dict({
                "x": batch.x,
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "batch": batch.batch
               })
               out = model(input_dict)
               val_loss = (out.squeeze() - batch.z).abs().mean()
               total_error = val_loss.item() * batch.num_nodes
               total_count = batch.num_nodes

       self.log("val_count", total_count, batch_size=config.batch_size, on_step=False, on_epoch=True, sync_dist=True)
       self.log("val_loss", total_error, batch_size=config.batch_size, on_step=False, on_epoch=True, sync_dist=True)
       return val_loss

    # activated before each epoch
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    
    # activated after each epoch
    def on_train_epoch_end(self):
        #receive "train_loss" and "train_count" of the current epoch
        train_loss = self.trainer.callback_metrics.get('train_loss',None)
        train_count = self.trainer.callback_metrics.get('train_count', None)
        #receive "validation_loss" and "validation_count" of the current epoch
        val_loss = self.trainer.callback_metrics.get('val_loss',None)
        val_count = self.trainer.callback_metrics.get('val_count', None)

        train_loss = train_loss / train_count
        val_loss = val_loss / val_count

        #epoch timer
        self.epoch_end_time = time.time()
        epoch_time = self.epoch_end_time - self.epoch_start_time
        
        scheduler = self.lr_schedulers()
        scheduler.step(val_loss)
        #display train_loss and val_loss on the head node
        if (torch.distributed.get_rank() == 0 and self.current_epoch % self.hparams.config.epoch_step == 0 ):
            print("\n\ntrain_loss: ",train_loss.item()," val_loss: ",val_loss.item()," elapsed_time: ",epoch_time,"\n")
            torch.save(
            {
                    "config": self.hparams.config,
                    "epoch": self.current_epoch,
                    "model_state_dict": self.chemmodel.state_dict(),
                    "loss": train_loss
            }, self.hparams.model_save_path)
         #display the optimizer information
         #print(self.optimizers().optimizer.state_dict)

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
    assert config.dataset in ["ENERGY_DATASET", "FORCES_DATASET"]
    data_type = None
    if config.dataset == 'ENERGY_DATASET':
        data_type = 'energy'
    elif config.dataset == 'FORCES_DATASET':
        data_type = 'force'

    dataset = MutilWaterDataset(root=dataset_path, split=f"Water_Round4_optimized_{data_type}")
    trainset_list.append(dataset)
    dataset = MutilWaterDataset(root=dataset_path, split=f"Water_Round4_optimized_test_{data_type}")
    valset_list.append(dataset)

    for i in range(1, 22):
        dataset = MutilWaterDataset(root=dataset_path, split=f"Water{i}_Round4_{data_type}")
        tr_data, v_data, te_data = split_dataset(dataset, train_p=config.train_ratio, val_p=config.val_ratio,
                                                 shuffle=True)
        trainset_list.append(tr_data)
        valset_list.append(v_data)

    for i in range(22, 31, 1):
        dataset = MutilWaterDataset(root=dataset_path, split=f"Water{i}_Round4_{data_type}")
        tr_data, v_data, te_data = split_dataset(dataset, train_p=config.train_ratio, val_p=config.val_ratio, shuffle=True)
        trainset_list.append(tr_data)
        valset_list.append(v_data)

    for i in range(40, 101, 10):
        dataset = MutilWaterDataset(root=dataset_path, split=f"Water{i}_Round4_{data_type}")
        tr_data, v_data, te_data = split_dataset(dataset, train_p=config.train_ratio, val_p=config.val_ratio, shuffle=True)
        trainset_list.append(tr_data)
        valset_list.append(v_data)

    for i in range(200, 1001, 100):
        dataset = MutilWaterDataset(root=dataset_path, split=f"Water{i}_Round4_{data_type}")
        tr_data, v_data, te_data = split_dataset(dataset, train_p=config.train_ratio, val_p=config.val_ratio, shuffle=True)
        trainset_list.append(tr_data)
        valset_list.append(v_data)

    train_dataset = ConcatDataset(trainset_list)
    val_dataset = ConcatDataset(valset_list)

    g = torch.Generator()
    g.manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)

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

    assert config.model in ["ChemGNN_energy", "ChemGNN_forces"]
    model = None
    if config.model == 'ChemGNN_energy':
        model = ENERGY_Model().to(config.device)
    elif config.model == 'ChemGNN_forces':
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

    # get the number of SLURM nodes
    nnodes = int(os.getenv("SLURM_NNODES"))
    # initiatie a ChemLightning object named pdmdlightning for parallel training
    pdmdlightning = ChemLightning(model,config,model_save_path)
    # set up a trainer for pdmdlightning
    trainer = lightning.Trainer(num_nodes=nnodes, strategy="ddp",accelerator="gpu",devices=1, max_epochs=config.epoch)
    trainer.fit(model=pdmdlightning, train_dataloaders=train_loader,val_dataloaders=val_loader)

    print("[Step 6] Saving final model...")

    final_state_dict = model.state_dict()
    torch.save(final_state_dict, final_model_state_path)
