# Quick test in Python
import torch
import schnetpack as spk
from schnorb.data import SchNOrbAtomsData

# Load your data
data = SchNOrbAtomsData('example_data/h2o_hamiltonians.db')
print(f"Dataset contains {len(data)} molecules")

# Load pretrained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('pretrained_models/model_100k_split.pt', map_location=device)
model.eval()

# Get a sample
sample = data[0]
print("Sample keys:", sample.keys())

# Make a prediction
with torch.no_grad():
    # Move sample to device and add batch dimension
    batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in sample.items()}
    prediction = model(batch)
    print("Prediction keys:", prediction.keys())