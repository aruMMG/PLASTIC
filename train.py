
import os
import csv
import torch
import yaml
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import io

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils.plot import plot_wrong
from utils.utils import createFolders, load_checkpoint
from Datasets.dataset import create_dataloader
from models.models import ModelFactory
from utils.loss import load_loss_from_config
from utils.optimizer import load_optimizer_from_config

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(args, class_names, start_epoch, model, optimizer, scheduler, loss_fn, train_dataloader,val_dataloader=None, writer=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_val_accuracy = 0.0
    model.to(device)

    for epoch in range(start_epoch, args.epochs):
        
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        epoch_start_time = time.time()

        for x_train, y_train in train_dataloader:
            
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_samples += x_train.shape[0] 
        
        scheduler.step()
        epoch_loss /= total_samples
        
        # Logging start
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)

        # Log gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar("Gradient Norm", total_norm, epoch)

        epoch_time = time.time() - epoch_start_time
        writer.add_scalar("Time/Epoch", epoch_time, epoch)
        # Logging ends

        if epoch % args.check_epoch == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"Weights/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

            print(f'saving checkpoint for epoch {epoch+1}')
            checkpoint = {
                            'epoch': epoch,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict() if scheduler else None,
                        }
            torch.save(checkpoint, f'logFile/{args.exp_name}/weights/checkpoint_{epoch}.pth')
            
            if val_dataloader is not None:
                model.eval()
                test_correct = 0
                test_size = 0
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                for x_val, y_val in val_dataloader:
                    test_size+=len(y_val)
                    x_val = x_val.to(device)
                    y_val = y_val.type(torch.LongTensor)
                    y_val = y_val.to(device)
                    with torch.no_grad():
                        outputs = model(x_val)
                        _, predicted = torch.max(outputs.data, 1)                        
                        test_correct += (predicted == y_val).sum().item()
                        val_loss += loss_fn(outputs, y_val).item()
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(y_val.cpu().numpy())
                test_accuracy = test_correct/test_size
                val_loss /= len(val_dataloader)

                writer.add_scalar("Accuracy/val", test_accuracy, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)

                cm = confusion_matrix(all_labels, all_preds)
                cm_figure = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
                cm_figure.figure_.savefig(f"logFile/{args.exp_name}/confusion_matrix_epoch_{epoch}.png")
                writer.add_figure("Confusion Matrix", cm_figure.figure_, epoch)

                # Log example predictions
                # sample_inputs = x_val[:4].cpu()  # Get a few sample inputs
                # sample_predictions = predicted[:4].cpu()
                # writer.add_images("Example Predictions", sample_inputs, epoch)
                # print(f"Example predictions logged for epoch {epoch}.")

                # print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Test accuracy = {test_accuracy:.4f}")
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Validation Loss = {val_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

                if best_val_accuracy<=test_accuracy:
                    best_val_accuracy = test_accuracy
                    best_epoch = epoch+1
                    print(f'saving checkpoint for epoch {epoch+1}')
                    checkpoint = {
                                    'epoch': epoch,
                                    'model_state': model.state_dict(),
                                    'optimizer_state': optimizer.state_dict(),
                                    'scheduler_state': scheduler.state_dict() if scheduler else None,
                                }
                    torch.save(checkpoint, f'logFile/{args.exp_name}/weights/best_checkpoint.pth')

    print(f'saving checkpoint for last')
    checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler else None,
                }
    torch.save(checkpoint, f'logFile/{args.exp_name}/weights/last_checkpoint.pth')


    if val_dataloader:
        with open(os.path.join("logFile",args.exp_name, f'Val_results_weights.csv'), 'a', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(["best_epoch", "Max_Correct"])
            writer_csv.writerow([best_epoch, best_val_accuracy])
        if args.plot:
            if args.plot_wrong:
                assert args.test_batch_size==1, f"plot wrong required test batch size of 1 but {args.test_batch_size} provided"
                plot_wrong(args, class_names, model, val_dataloader, f'logFile/{args.exp_name}/weights/best_checkpoint.pth')

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = createFolders(args)
    args.exp_name = exp_name

    with open(args.yaml, 'r') as file:
        config = yaml.safe_load(file)

    writer = SummaryWriter(log_dir=f'logFile/{args.exp_name}/tensorboard')

    # Dataloader
    train_path = config['train']
    if "val" in config:
        val_path = config['val']
    else:
        val_path = config["train"]
    class_names = config["class_names"]
    print(class_names)

    train_loader, config = create_dataloader(config, train_path, class_names, args.batch_size, FC=args.FC, drop_last=True)
    val_loader, _ = create_dataloader(config, val_path, class_names, args.test_batch_size, FC=args.FC, shuffle=False)

    # Model
    factory = ModelFactory()
    model = factory.load_model_from_yaml(config)    
    model.to(device)

    optimizer = load_optimizer_from_config(config, model.parameters())
    print(f'original optimiser learning rate: {get_lr(optimizer)}')
    
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    start_epoch = 0
    print(f'after scheduler learning rate: {get_lr(optimizer)}')

    if args.resume:
        start_epoch = load_checkpoint(args, device, model, optimizer, scheduler)

    loss_fn = load_loss_from_config(config, device, val_loader)

    # Training
    train(args, class_names, start_epoch, model, optimizer, scheduler, loss_fn, train_loader, val_loader, writer)
    writer.close()

def update_args_with_hyp(args):
    """
    Update args with hyperparameters from a YAML file.
    Adds new arguments if they are in the YAML file but not in the argparse object.
    """
    if args.hyp:
        with open(args.hyp, 'r') as file:
            hyp_params = yaml.safe_load(file)
            for key, value in hyp_params.items():
                setattr(args, key, value)  # Add or override arguments dynamically
    return args


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    # parser.add_argument('--exp_name', type=str, default='NoName', help='Save log file name')
    # parser.add_argument('--resume', type=str, default=None, help='provide checkpoint path if resume')
    # parser.add_argument('--yaml', type=str, default='yaml/Leone_trans_yaml.yaml', help='input data directoy containing .npy files')
    # parser.add_argument('--hyp', type=str, default='yaml/Leone_trans_hyp.yaml', help='input data directoy containing .npy files')
        
    # parser.add_argument('--plot', action='store_true', help='plot fake and real dataa examples if true')
    # parser.add_argument('--plot_wrong', action='store_true', help='plot data when predicted wrong if true')

    parser.add_argument('--exp_name', type=str, default='NoName', help='Save log file name')
    parser.add_argument('--resume', type=str, default=None, help='provide checkpoint path if resume')
    parser.add_argument('--yaml', type=str, default='yaml/Leone/Leone_5class_Improved_Incep.yaml', help='input data directoy containing .npy files')
    parser.add_argument('--hyp', type=str, default='yaml/Leone/Leone_Incep_hyp.yaml', help='input data directoy containing .npy files')
        
    parser.add_argument('--plot', action='store_true', help='plot fake and real dataa examples if true')
    parser.add_argument('--plot_wrong', action='store_true', help='plot data when predicted wrong if true')

    args = parser.parse_args()
    args = update_args_with_hyp(args)

    main(args)

    

