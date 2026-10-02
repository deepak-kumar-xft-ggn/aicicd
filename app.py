# Databricks notebook source
from flask import Flask, request, jsonify
import torch, torch.nn as nn

app = Flask(__name__)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = torch.tensor(request.json["input"])
    output = model(data.float())
    _, predicted = torch.max(output, 1)
    return jsonify({"prediction": predicted.item()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)