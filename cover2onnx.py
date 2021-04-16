# _*_ coding: utf-8 _*_ 
import torch
from core.mobilefacenet import MobileFacenet


def pytorch2onnx(model_path, save_path, input_shape):
    # load model
    model = MobileFacenet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # an example input you would noremally previde to your model's forward() method
    x = torch.rand(input_shape)
    #Export the model
    torch_out = torch.onnx._export(model, x, save_path, export_params=True)

if __name__ == "__main__":
    model_path = "models_3/model_1_120.pth"
    save_path = "models_3/model_1_120.onnx"
    input_shape = (1,3,112,112)
    pytorch2onnx(model_path, save_path, input_shape)
