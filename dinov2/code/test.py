from models.vision_transformer import vit_base
import torch
if __name__ == "__main__":
    model = vit_base(patch_size=16, num_register_tokens=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).half()
    input = torch.randn(1, 3, 512, 512).to(device).half()
    output = model(input)
    print(output.shape)