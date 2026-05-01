import torch
import torch.nn as nn
import numpy as np
from models import FNO, U_net
import wandb
import json
from torch.utils.data import  DataLoader
from dataset import KolmogorovDataset

# Hyperparameter space for the sweep
sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'architecture': {'values': ['FNO', 'U-Net']},
        'width': {'values': [32, 64]},
        'modes': {'values': [12, 20]},
        'learning_rate': {'values': [1e-3, 5e-4]},
        'batch_size': {'value': 64},
        'epochs': {'value': 20}
    }
}

def nrmse_loss(pred, target, eps=1e-8):
    diff_norm = torch.norm(pred - target, p=2, dim=(2, 3)) 
    target_norm = torch.norm(target, p=2, dim=(2, 3))
    return torch.mean(diff_norm / (target_norm + eps))

def train():

    with open("preprocessing_info.json") as f:
        preprocessing_data = json.load(f)

    with wandb.init() as run:
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.architecture == "FNO":
            run_name = f"FNO_w{config.width}_m{config.modes}"
        else:
            run_name = f"UNet_w{config.width}"
        
        run.name = run_name
        
        dataset_path = preprocessing_data["dataset_path"]
        train_sim_idx = preprocessing_data["train_idx"]
        test_sim_idx = preprocessing_data["test_idx"]
        g_min = preprocessing_data["g_min"]
        g_max = preprocessing_data["g_max"]

        train_data = KolmogorovDataset(dataset_path, train_sim_idx, g_min, g_max)
        test_data = KolmogorovDataset(dataset_path, test_sim_idx, g_min, g_max)

        train_loader = DataLoader(
            train_data, 
            batch_size=wandb.config.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )

        val_loader = DataLoader(
            test_data, 
            batch_size=wandb.config.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
)
        # Select model
        if config.architecture == "FNO":
            model = FNO(modes1=config.modes, modes2=config.modes, width=config.width).to(device)
        else:
            model = U_net(width=config.width).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0
            for train_batch_x, train_batch_y in train_loader:
                train_batch_x, train_batch_y = train_batch_x.to(device), train_batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(train_batch_x)
                loss = criterion(output, train_batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_mse = 0
            val_nrmse = 0
            
            with torch.no_grad():
                for i, (validation_batch_x, validation_batch_y) in enumerate(val_loader):
                    validation_batch_x, validation_batch_y = validation_batch_x.to(device), validation_batch_y.to(device)
                    v_out = model(validation_batch_x)
                    
                    val_mse += criterion(v_out, validation_batch_y).item()
                    val_nrmse += nrmse_loss(v_out, validation_batch_y).item()
            
            avg_val_mse = val_mse / len(val_loader)
            avg_val_nrmse = val_nrmse / len(val_loader)

            # Log a WandB
            wandb.log({
                "epoch": epoch,
                "train_loss_mse": avg_train_loss,
                "val_loss_mse": avg_val_mse,
                "val_nrmse": avg_val_nrmse
            })
            
            print(f"Epoch {epoch} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_mse:.6f} | Val NRMSE: {avg_val_nrmse:.4f}")

        stats = {
            "g_min": g_min,
            "g_max": g_max,
            "dataset_path": dataset_path
        }
        if config.architecture == "U-Net":
            checkpoint_name = f"cp_{config.architecture}_w{config.width}_lr{config.learning_rate}.pth"
        else:
            checkpoint_name = f"cp_{config.architecture}_w{config.width}_m{config.modes}_lr{config.learning_rate}.pth"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': dict(config),
            'stats': stats 
        }
        
        torch.save(checkpoint, checkpoint_name)
        print(f"Model saved as: {checkpoint_name}")

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="kolmogorov-sweep")
    
    wandb.agent(sweep_id, function=train)