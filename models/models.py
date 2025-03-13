import yaml
import torch
from models.transformer import SpectroscopyTransformerEncoder_PreT
from models.FC import FCNet
from models.PSDN import ResidualNet, InceptionNetwork_PreT
from models.CNN import InceptionWithSE_PreT
from models.CNN1 import ImprovedInception_PreT
from models.compared_models import NNH10, NNHout, CNN

class ModelFactory:
    """
    A factory class to initialize and manage all models.
    """

    def __init__(self):
        self.models = {}

    def initialize_model(self, model_name, *args, **kwargs):
        """
        Initialize a model based on the given model name.
        :param model_name: Name of the model (e.g., 'Transformer', 'NC', 'PSDN')
        :param args: Positional arguments for the model
        :param kwargs: Keyword arguments for the model
        :return: An instance of the specified model
        """
        model_mapping = {
            'Transformer': SpectroscopyTransformerEncoder_PreT,
            'ANN': FCNet,
            'ANN1': NNH10,
            'ANN2': NNHout,
            'CNN': CNN,
            'PSDN Residual': ResidualNet,
            'PSDN Inception': InceptionNetwork_PreT,
            'CNN Inception': InceptionWithSE_PreT,
            'Improved Inception': ImprovedInception_PreT,
        }

        if model_name not in model_mapping:
            raise ValueError(f"Model '{model_name}' is not recognized. Available models: {list(model_mapping.keys())}")

        model_class = model_mapping[model_name]
        model_instance = model_class(*args, **kwargs)
        self.models[model_name] = model_instance

        return model_instance

    def get_model(self, model_name):
        """
        Retrieve an already initialized model.
        :param model_name: Name of the model
        :return: The model instance
        """
        return self.models.get(model_name, None)

    def load_model_from_yaml(self, config):
        """
        :param config: configuration from a YAML file
        :return: The initialized model
        """

        if 'model_name' not in config:
            raise ValueError("YAML file must contain a 'model_name' field.")

        model_name = config['model_name']
        model_args = config.get('model_arguments', {})

        model = self.initialize_model(model_name, **model_args)

        if model_name == 'PSDN Inception' or model_name=="CNN":
            dummy_input = torch.randn(32, 1, model_args["input_size"])
            model.initialize_fc1(dummy_input)

        return model

if __name__=="__main__":
    # Example usage in train.py
    factory = ModelFactory()
    # transformer_model = factory.initialize_model('Transformer', input_size=4000, num_classes=2, num_transformer_layers=3, mlp_size=64, patch_size=20, embedding_dim=20, num_heads=4)
    # nc_model = factory.initialize_model('ANN', 4000, num_class=2)
    # psdn_Inception = factory.initialize_model('PSDN Inception', 2)
    # psdn_Residual = factory.initialize_model('PSDN Residual', 2)
    

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_data = torch.randn(32, 1, 4000).to(device)
    model = factory.load_model_from_yaml("yaml/transformer_model.yaml")
    model.to(device)
    model.eval()
    output = model(input_data)
    print(f'model output: {output.shape}')

    # psdn_Residual.to(device)
    # psdn_Residual.eval()
    # output = psdn_Residual(input_data)
    # print(f'Inception model output: {output.shape}')

    # psdn_Inception.to(device)
    # psdn_Inception.eval()
    # output = psdn_Inception(input_data)
    # print(f'ResidualNet model output: {output.shape}')


