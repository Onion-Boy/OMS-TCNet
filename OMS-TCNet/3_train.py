import numpy as np
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.evaluation.metric import dice
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# data_dir = "./data/fullres/train"

# logdir = f"./logs/segmambav2"

# env = "DDP"
# model_save_path = os.path.join(logdir, "model")
# max_epoch = 1000
# batch_size = 2
# val_every = 2
# num_gpus = 4
# device = "cuda:0"
# patch_size = [128, 128, 128]
# augmentation = True 


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12
        
        from models.MSTCNet.mstcnet_brats import mstcnet
        self.model = mstcnet(in_channels=1, out_channels=5, img_size=(128, 128, 128), feature_size=32, res_block=True, dropout_rate=0.0)
                
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        
        self.loss_func = nn.CrossEntropyLoss()
      
    # def convert_labels(self, labels):
    #     ## 尾状核, 壳核, 外苍白球, 内苍白球
    #     # 1, 2, 3, 4  
    #     result = [(labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 1), (labels == 2), (labels == 3), (labels == 4)]
        
    #     return torch.cat(result, dim=1).float()

    def training_step(self, batch):
        import time 
        image, label = self.get_input(batch)

        pred = self.model(image)
        loss = self.loss_func(pred, label)

        self.log("train_loss", loss, step=self.global_step)
        return loss 

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()

        return image, label 

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        # 预测值和标签中都存在目标体素
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50]) # 第二个数字是个占位符（比如用于与其他指标配合，或未来拓展）
        # 都为背景
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        # 预测或标签中有一个是空，另一个不是
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.model(image).argmax(dim=1) # 在类别维度（dim=1）上取最大值的索引，model(image)是(batch_size, num_classes, D, H, W)，表示每个体素（voxel）在5个类别上的预测概率/得分
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 5
        for i in range(1, c):
            pred_c = output == i
            target_c = target == i

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        dices_mean = []
        c = 4 
        for i in range(0, c):
            dices_mean.append(dices[i].mean())

        mean_dice = sum(dices_mean) / len(dices_mean)
        
        self.log("1", dices_mean[0], step=self.epoch)
        self.log("2", dices_mean[1], step=self.epoch)
        self.log("3", dices_mean[2], step=self.epoch)
        self.log("4", dices_mean[3], step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)

        print("*" * 50)
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            path = os.path.join(model_save_path, f"best_model.pth")
            save_new_model_and_delete_last(self.model, 
                                            path, 
                                            delete_symbol="best_model")
            print("Save best model!!!  Path: {}".format(path))

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model.pth"), 
                                        delete_symbol="final_model")
        print(f"epoch {self.epoch} mean_dice is {mean_dice}, best_mean_dice is {self.best_mean_dice}")
        # print(f"mean_dice is {mean_dice}")


# logdir = f"./logs/segmambav2"
parameter_save_path = "/media/ly/LISA_Task2b/parameters/MSTCNet"
env = "DDP"
# model_save_path = os.path.join(logdir, "model")
model_save_path = os.path.join(parameter_save_path)
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 2
device = "cuda:0"
patch_size = [128, 128, 128]
augmentation = True 


if __name__ == "__main__":
    data_dir = "/media/ly/LISA_Task2b/train_fullres_process"
    
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            # logdir=logdir,
                            logdir=parameter_save_path,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17754,
                            training_script=__file__)

    from data.test_list import test_list
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    resume_path = "/media/ly/LISA_Task2b/parameters/MSTCNet/best_model.pth"  

    if os.path.exists(resume_path):
        trainer.load_state_dict(resume_path)
        print(f"Loading pretrained weights from {resume_path}")

    else:
        print(f"No pretrained weights found at {resume_path}, training from scratch.")
        
    trainer.train(train_dataset=train_ds, val_dataset=test_ds)
    # End Child Processes
    # pkill -f train.py
