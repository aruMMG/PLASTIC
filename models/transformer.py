import torch.nn as nn
import torch
import torch.nn.functional as F

# from common import PreBlock
from models.common import PreBlock

class PatchEmbedding(nn.Module):
    """
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=1,
                 patch_size:int=20,
                 embedding_dim:int=20):
        super().__init__()
        
        self.patch_size = patch_size
        self.patcher = nn.Conv1d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        # 3. Create a layer to turn an image into patches
        # self.patcher = nn.Conv2d(in_channels=in_channels,
        #                          out_channels=embedding_dim,
        #                          kernel_size=patch_size,
        #                          stride=patch_size,
        #                          padding=0)

    # 5. Define the forward method 
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        # print(x_patched.shape)
        # 6. Make sure the output shape has the right order 
        return x_patched.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, 

class SpectroscopyTransformerEncoder(nn.Module): 
  def __init__(self,
               input_size=4000, # from Table 3
               num_channels=1,
               patch_size=20,
               embedding_dim=20, # from Table 1
               dropout=0.1, 
               mlp_size=256, # from Table 1
               num_transformer_layers=3, # from Table 1
               num_heads=4, # from Table 1 (number of multi-head self attention heads)
               num_classes=2): # generic number of classes (this can be adjusted)
    super().__init__()

    # Assert image size is divisible by patch size 
    assert input_size % patch_size == 0, "Image size must be divisble by patch size."

    # 1. Create patch embedding
    self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

    # 2. Create class token
    self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                    requires_grad=True)

    # 3. Create positional embedding
    num_patches = input_size // patch_size
    self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))

    # 4. Create patch + position embedding dropout 
    self.embedding_dropout = nn.Dropout(p=dropout)

    # # 5. Create Transformer Encoder layer (single)
    # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
    #                                                             nhead=num_heads,
    #                                                             dim_feedforward=mlp_size,
    #                                                             activation="gelu",
    #                                                             batch_first=True,
    #                                                             norm_first=True)

    # 5. Create stack Transformer Encoder layers (stacked single layers)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                              nhead=num_heads,
                                                                                              dim_feedforward=mlp_size,
                                                                                              activation="gelu",
                                                                                              batch_first=True,
                                                                                              norm_first=True), # Create a single Transformer Encoder Layer
                                                     num_layers=num_transformer_layers) # Stack it N times

    # 7. Create MLP head
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )

  def forward(self, x):
    # Get some dimensions from x
    batch_size = x.shape[0]

    # Create the patch embedding
    x = self.patch_embedding(x)
    # print(x.shape)

    # First, expand the class token across the batch size
    class_token = self.class_token.expand(batch_size, -1, -1) # "-1" means infer the dimension

    # Prepend the class token to the patch embedding
    x = torch.cat((class_token, x), dim=1)
    # print(x.shape)

    # Add the positional embedding to patch embedding with class token
    x = self.positional_embedding + x
    # print(x.shape)

    # Dropout on patch + positional embedding
    x = self.embedding_dropout(x)

    # Pass embedding through Transformer Encoder stack
    x = self.transformer_encoder(x)

    # Pass 0th index of x through MLP head
    x = self.mlp_head(x[:, 0])

    return x




class SpectroscopyTransformerEncoder_PreT(nn.Module):
    """
    fNIRS-PreT model

    Args:
        n_class: number of classes.
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
        dim: last dimension of output tensor after linear transformation.
        depth: number of Transformer blocks.
        heads: number of the multi-head self-attention.
        mlp_dim: dimension of the MLP layer.
        pool: MLP layer classification mode, 'cls' is [CLS] token pooling, 'mean' is  average pooling, default='cls'.
        dim_head: dimension of the multi-head self-attention, default=64.
        dropout: dropout rate, default=0.
        emb_dropout: dropout for patch embeddings, default=0.
    """
    def __init__(self,
               input_size=4000, # from Table 3
               num_channels=1,
               patch_size=20,
               embedding_dim=20, # from Table 1
               dropout=0.1, 
               mlp_size=256, # from Table 1
               num_transformer_layers=3, # from Table 1
               num_heads=4, # from Table 1 (number of multi-head self attention heads)
               num_classes=2,
               pre_module=False): # generic number of classes (this can be adjusted)
        super().__init__()
        self.pre_module = pre_module
        self.pre = PreBlock(input_size=input_size)
        self.IR_PreT = SpectroscopyTransformerEncoder(input_size, num_channels, patch_size, embedding_dim, dropout, mlp_size, num_transformer_layers, num_heads, num_classes)


    def forward(self, x):
        if self.pre_module:
            x = self.pre(x)
        x = self.IR_PreT(x)
        return x

    

if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    input_data = torch.randn(batch_size, 1, 4000).to(device)
    model = SpectroscopyTransformerEncoder_PreT(num_classes=9, pre_module=True)
    model.to(device)
    model.eval()
    total_time = 0.0
    with torch.no_grad():
        for i in range(1000):
            time1 = time.time()
            output = model(input_data)
            time2 = time.time()
            time_per_data = time2-time1
            total_time+=time_per_data
    print(f"time taken is :", total_time)
    print(f"time taken is avarage:", total_time/1000)
    print(f'Inception model output: {output.shape}')
    assert output.shape == (batch_size,9), "Output shape is incorrect."