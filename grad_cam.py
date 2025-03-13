import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import ModelFactory
from Datasets.dataset import create_dataloader
from c3 import heatmap_plot
def target_category_loss(x, category_index, nb_classes):
    return torch.mul(x, F.one_hot(category_index, nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        # print(output.size())
        return output[target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            output = output.squeeze()
            target_category = np.argmax(output.cpu().data.numpy())
            # print(output)
            # print(target_category)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        #weights = np.mean(grads, axis=(0))
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
             cam += w * activations[i, :]
        # cam = activations.T.dot(weights)
        # cam = activations.dot(weights)
        # cam = activations.dot(weights)
        # print(input_tensor.shape[1])
        # print(cam.shape)
        # x = np.arange(0, 247, 1)
        # plt.plot(x, cam.reshape(-1, 1))
        # sns.set()
        # ax = sns.heatmap(cam.reshape(-1, 1).T)
        #cam = cv2.resize(cam, input_tensor.shape[1:][::-1])
        #cam = resize_1d(cam, (input_tensor.shape[2]))
        cam = np.interp(np.linspace(0, cam.shape[0], input_tensor.shape[2]), np.linspace(0, cam.shape[0], cam.shape[0]), cam)   #Change it to the interpolation algorithm that numpy comes with.
        #cam = np.maximum(cam, 0)
        # cam = np.expand_dims(cam, axis=1)
        # ax = sns.heatmap(cam)
        # plt.show()
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)#归一化处理
        # heatmap = (cam - np.mean(cam, axis=-1)) / (np.std(cam, axis=-1) + 1e-10)
        print(heatmap.shape)
        return heatmap
class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=1)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=1)
        return weights
# from pytorch_grad_cam.utils.image import preprocess_image
import matplotlib.pyplot as plt
def plot_and_save(data1, data2, filename='plot.png'):
    """
    Plots two one-dimensional data series on the same plot and saves it as an image.

    Parameters:
        data1 (list or numpy array): First one-dimensional data series.
        data2 (list or numpy array): Second one-dimensional data series.
        filename (str): Filename to save the plot image. Default is 'plot.png'.
    """
    # Create a new figure
    fig, ax1 = plt.subplots()

    # Plot data1
    ax1.plot(data1, label='Data 1', color='tab:blue')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Data 1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for data2
    ax2 = ax1.twinx()
    ax2.plot(data2, label='Data 2', color='tab:red')
    ax2.set_ylabel('Data 2', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Save the plot as an image
    plt.savefig(filename)

    # Close the plot to release resources
    plt.close()


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
    import os
    import yaml
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('--exp_name', type=str, default='logFile/Edi_MIR/Edi_MIR_Improved_Incep/', help='Experiment name for logging results')
    parser.add_argument('--checkpoint', type=str, default="logFile/Edi_MIR/baseline/Edi_MIR_trans/Edi_MIR_trans_1/weights/best_checkpoint.pth", help='Path to the model checkpoint')
    parser.add_argument('--yaml', type=str, default="yaml/Edi_MIR/trans/trans_pre.yaml", help='Path to the model configuration YAML file')
    parser.add_argument('--hyp', type=str, default="yaml/Edi_MIR/trans/trans_pre_hyp.yaml", help='input data directoy containing .npy files')
    parser.add_argument('--save_path', type=str, default='.logFile/Edi_MIR/baseline/Edi_MIR_trans/Edi_MIR_trans_1/test/heatmap/', help='input data directoy containing .npy files')
    args = parser.parse_args()
    args = update_args_with_hyp(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model configuration
    with open(args.yaml, 'r') as file:
        config = yaml.safe_load(file)
    # Dataloader
    test_path = config['test']
    class_names = config["class_names"]
    print(class_names)
    test_loader, config = create_dataloader(config=config, dir=test_path, class_names=class_names, bs=args.test_batch_size, FC=args.FC, shuffle=False)

    # Load model
    model_args = config.get('model_arguments', {})
    factory = ModelFactory()
    model = factory.load_model_from_yaml(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    # for Improved Inception
    target_layer = model.IR_PreT.inception2
    # for PSDN
    # target_layer = model.IR_PreT.inception1
    # for trans
    # target_layer = model.IR_PreT.transformer_encoder
    
    # model = InceptionNetwork_PreT(4)
    # model.load_state_dict(torch.load(args.weight)['model'])
    # target_layer = model.IR_PreT.inception1

    # model = SpectroscopyTransformerEncoder_PreT(num_classes=4, mlp_size=64)
    # model.load_state_dict(torch.load(args.weight)['model'])
    # target_layer = model.IR_PreT.transformer_encoder
    
    # model = SpectroscopyTransformerEncoder(num_classes=4, mlp_size=64)
    # model.load_state_dict(torch.load(args.weight)['model'])
    # target_layer = model.transformer_encoder
    
    
    net = GradCAM(model, target_layer)
  





    # test(args, model, val_dataloader, weight_path, n_classes=len(np.unique(dataarrayY_val)), arch=args.model)

    heatmap0, heatmap1, heatmap2, heatmap3 = None, None, None, None
    heatmap4, heatmap5, heatmap6, heatmap7, heatmap8 = None, None, None, None, None
    input_tensors0, input_tensors1, input_tensors2, input_tensors3 = None, None, None, None
    input_tensors4, input_tensors5, input_tensors6, input_tensors7, input_tensors8 = None, None, None, None, None
    for input_tensor, labels in test_loader:
        # inputs, labels = inputs.to(device), labels.to(device)
        # outputs = model(inputs)
        # _, predicted = torch.max(outputs.data, 1)
        output = net(input_tensor)
        input_tensor1 = input_tensor.numpy().squeeze()

        if labels==0:        
            if heatmap0 is None:
                heatmap0 = output
                input_tensors0 = input_tensor1
            else:
                heatmap0 = np.vstack([heatmap0, output])
                input_tensors0 = np.vstack([input_tensors0, input_tensor1])
        if labels==1:        
            if heatmap1 is None:
                heatmap1 = output
                input_tensors1 = input_tensor1
            else:
                heatmap1 = np.vstack([heatmap1, output])
                input_tensors1 = np.vstack([input_tensors1, input_tensor1])
        if labels==2:        
            if heatmap2 is None:
                heatmap2 = output
                input_tensors2 = input_tensor1
            else:
                heatmap2 = np.vstack([heatmap2, output])
                input_tensors2 = np.vstack([input_tensors2, input_tensor1])
        if labels==3:        
            if heatmap3 is None:
                heatmap3 = output
                input_tensors3 = input_tensor1
            else:
                heatmap3 = np.vstack([heatmap3, output])
                input_tensors3 = np.vstack([input_tensors3, input_tensor1])
        if labels==4:        
            if heatmap4 is None:
                heatmap4 = output
                input_tensors4 = input_tensor1
            else:
                heatmap4 = np.vstack([heatmap4, output])
                input_tensors4 = np.vstack([input_tensors4, input_tensor1])
        if labels==5:        
            if heatmap5 is None:
                heatmap5 = output
                input_tensors5 = input_tensor1
            else:
                heatmap5 = np.vstack([heatmap3, output])
                input_tensors5 = np.vstack([input_tensors5, input_tensor1])
        if labels==6:        
            if heatmap6 is None:
                heatmap6 = output
                input_tensors6 = input_tensor1
            else:
                heatmap6 = np.vstack([heatmap6, output])
                input_tensors6 = np.vstack([input_tensors6, input_tensor1])
        if labels==7:        
            if heatmap7 is None:
                heatmap7 = output
                input_tensors7 = input_tensor1
            else:
                heatmap7 = np.vstack([heatmap7, output])
                input_tensors7 = np.vstack([input_tensors7, input_tensor1])
        if labels==8:        
            if heatmap8 is None:
                heatmap8 = output
                input_tensors8 = input_tensor1
            else:
                heatmap8 = np.vstack([heatmap8, output])
                input_tensors8 = np.vstack([input_tensors8, input_tensor1])
    if not os.path.exists(args.save_path):
        os.mkdir(f"{args.save_path}")
    np.save(f"{args.save_path}/heatmap0.npy", heatmap0)
    np.save(f"{args.save_path}/heatmap1.npy", heatmap1)
    np.save(f"{args.save_path}/heatmap2.npy", heatmap2)
    np.save(f"{args.save_path}/heatmap3.npy", heatmap3)
    np.save(f"{args.save_path}/heatmap4.npy", heatmap3)
    np.save(f"{args.save_path}/heatmap5.npy", heatmap3)
    np.save(f"{args.save_path}/heatmap6.npy", heatmap3)
    np.save(f"{args.save_path}/heatmap7.npy", heatmap3)
    np.save(f"{args.save_path}/heatmap8.npy", heatmap3)
    np.save(f"{args.save_path}/input_tensors0.npy", input_tensors0)
    np.save(f"{args.save_path}/input_tensors1.npy", input_tensors1)
    np.save(f"{args.save_path}/input_tensors2.npy", input_tensors2)
    np.save(f"{args.save_path}/input_tensors3.npy", input_tensors3)
    np.save(f"{args.save_path}/input_tensors4.npy", input_tensors3)
    np.save(f"{args.save_path}/input_tensors5.npy", input_tensors3)
    np.save(f"{args.save_path}/input_tensors6.npy", input_tensors3)
    np.save(f"{args.save_path}/input_tensors7.npy", input_tensors3)
    np.save(f"{args.save_path}/input_tensors8.npy", input_tensors3)

    # This value for Leone dataset
    selected_indices0 = [109, 130, 112, 163, 23, 175, 139, 157, 30, 1]
    selected_indices1 = [226, 184, 253, 275, 239, 276, 89, 68, 94, 91]
    selected_indices2 = [175, 7, 191, 182, 92, 41, 12, 183, 107, 187]
    selected_indices3 = [3, 104, 73, 28, 20, 2, 67, 68, 109, 88]
    selected_indices4 = [5, 0, 7, 9, 13, 14, 3, 1, 4, 8]
    # This value for Jeon dataset
    # selected_indices0 = [86, 179, 332, 2, 320, 12, 21, 44, 256, 458]
    # selected_indices1 = [118, 249, 379, 555, 514, 377, 387, 571, 657, 138]
    # selected_indices2 = [332, 225, 361, 202, 223, 181, 203, 205, 294, 351]
    # selected_indices3 = [20, 52, 738, 470, 446, 722, 615, 707, 563, 169]
    # selected_indices4 = [104, 269, 63, 222, 228, 267, 22, 65, 68, 136]
    # selected_indices = heatmap_plot(heatmap0, input_tensors0, os.path.join(args.save_path, "0"))

    heatmap_plot(heatmap0, input_tensors0, os.path.join(args.save_path, "0"), selected_indices=selected_indices0)
    heatmap_plot(heatmap1, input_tensors1, os.path.join(args.save_path, "1"), selected_indices=selected_indices1)
    heatmap_plot(heatmap2, input_tensors2, os.path.join(args.save_path, "2"), selected_indices=selected_indices2)
    heatmap_plot(heatmap3, input_tensors3, os.path.join(args.save_path, "3"), selected_indices=selected_indices3)
    heatmap_plot(heatmap4, input_tensors4, os.path.join(args.save_path, "4"), selected_indices=selected_indices4)
    # heatmap_plot(heatmap5, input_tensors5, os.path.join(args.save_path, "5"), selected_indices=selected_indices)
    # heatmap_plot(heatmap6, input_tensors6, os.path.join(args.save_path, "6"), selected_indices=selected_indices)
    # heatmap_plot(heatmap7, input_tensors7, os.path.join(args.save_path, "7"), selected_indices=selected_indices)
    # heatmap_plot(heatmap8, input_tensors8, os.path.join(args.save_path, "8"), selected_indices=selected_indices)
    # selected_indices = heatmap_plot(heatmap0, input_tensors0, os.path.join(args.save_path, "0"))
    # selected_indices = heatmap_plot(heatmap1, input_tensors1, os.path.join(args.save_path, "1"))
    # selected_indices = heatmap_plot(heatmap2, input_tensors2, os.path.join(args.save_path, "2"))
    # selected_indices = heatmap_plot(heatmap3, input_tensors3, os.path.join(args.save_path, "3"))
    # selected_indices = heatmap_plot(heatmap4, input_tensors4, os.path.join(args.save_path, "4"))
    # selected_indices = heatmap_plot(heatmap5, input_tensors5, os.path.join(args.save_path, "5"))
    # selected_indices = heatmap_plot(heatmap6, input_tensors6, os.path.join(args.save_path, "6"))
    # selected_indices = heatmap_plot(heatmap7, input_tensors7, os.path.join(args.save_path, "7"))
    # selected_indices = heatmap_plot(heatmap8, input_tensors8, os.path.join(args.save_path, "8"))


