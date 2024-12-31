import torch
from dqn_agent import DQNNetwork  # Ensure this import matches the location of your DQNNetwork class
import netron

# Initialize the network
input_dim = 8
output_dim = 2
model = DQNNetwork(input_dim, output_dim)

# Export to ONNX
dummy_input = torch.randn(1, input_dim)
torch.onnx.export(model, dummy_input, "dqn_model.onnx")

# Visualize with Netron
netron.start("dqn_model.onnx")