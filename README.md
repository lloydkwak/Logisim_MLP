# Logisim_MLP

A complete hardware implementation of a multilayer perceptron (MLP) neural network for MNIST handwritten digit recognition, built with Logisim Evolution.

## Overview

This project demonstrates a fully digital circuit that classifies 16×16 binarized MNIST images (0–9) in real time. The network comprises:
- **Input layer:** 256 inputs (16×16 pixels)  
- **Hidden layer:** 128 neurons with ReLU activation  
- **Output layer:** 10 neurons for digit classes 0–9  
- **Total parameters:** ~34,000 weights  

All layers and activation functions are realized in Logisim Evolution, with trained weights stored in ROM modules.

## Repository Structure

Logisim_MLP/
├── MLP_circuit.circ # Main Logisim circuit file
├── weights/ # Generated weight and bias files
│ ├── layer1_weights.txt
│ ├── layer1_bias.txt
│ ├── layer2_weights.txt
│ └── layer2_bias.txt
├── python/ # Python scripts for weight generation
│ └── generate_weights.py
└── README.md # This file


## Requirements

- **Logisim Evolution** ≥ 3.8.0  
- **Java** ≥ 8 (for Logisim)  
- **Python** ≥ 3.8 (for weight generation)

## Generating Weights

1. Install Python dependencies (if any):
pip install numpy

2. Run the weight-generation script:
cd python
python generate_weights.py

3. Confirm the output files in `weights/`:
- `layer1_weights.txt`  
- `layer1_bias.txt`  
- `layer2_weights.txt`  
- `layer2_bias.txt`  

## Loading the Circuit

1. Open **Logisim Evolution**.  
2. Go to **File → Open** and select `MLP_circuit.circ`.  
3. Expand the **HiddenLayer** subcircuit, right-click the weight ROM, choose **Load Image**, and load:
- `weights/layer1_weights.txt`  
- `weights/layer1_bias.txt`  
4. Repeat for the **OutputLayer**:
- `weights/layer2_weights.txt`  
- `weights/layer2_bias.txt`  

## Simulation Setup

1. In Logisim, enable ticks: **Simulate → Ticks Enabled** (Ctrl+K).  
2. Set clock frequency: **Simulate → Tick Frequency → 1 kHz or higher**.  

## Testing & Debugging

1. Start simulation (Ctrl+K).  
2. Use the on-screen joystick to draw a digit on the RGB video input.  
3. Set the **start** input to `1` to trigger classification.  
4. After ~30–40 s of simulated time, the predicted digit appears on the 7-segment display.  

### Step-by-Step Debug

- Reduce frequency to 1 Hz for single-step inspection.  
- Use the probe tool to monitor:
- **InputLayer** outputs (256 pixel values)  
- **HiddenLayer** outputs (128 neuron activations)  
- **OutputLayer** scores (10 class logits)  

## Author

**Lloyd Kwak**  
GitHub: [lloydkwak](https://github.com/lloydkwak)

## License

This project is released under the MIT License. Feel free to use and modify for educational purposes.
