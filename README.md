```markdown
# Deep Learning Notebooks for XOR and MNIST

This repository contains three Jupyter notebooks demonstrating various deep learning techniques using PyTorch. Each notebook focuses on a different aspect of model design, hyperparameter tuning, and real-world datasets (XOR and MNIST).

---

## Repository Overview

1. **PyTorch XOR Implementation and Analysis**  
   - Implements a simple neural network to solve the classic XOR problem.  
   - Demonstrates how to configure hidden layers, weights, and activation functions.  
   - Explores **bypass connections** and minimal neuron architectures to achieve zero training loss.

2. **Hyperparameter Exploration for XOR**  
   - Extends the XOR model to test multiple hyperparameters (learning rates, hidden units, bypass).  
   - Tracks **convergence speed**, failure counts, and best configurations through early stopping.  
   - Shows how **Agile experimentation** with PyTorch can greatly influence model stability.

3. **Deep Learning on MNIST: From Simple Networks to CNNs**  
   - Demonstrates fully connected and convolutional neural networks for MNIST digit classification.  
   - Achieves **>99% validation accuracy** using logistic regression, multi-layer perceptrons, and CNN architectures.  
   - Visualizes learned **filters and feature maps** to highlight how CNNs capture spatial patterns.

---

## Requirements

- Python 3.x  
- [PyTorch](https://pytorch.org/)  
- NumPy, Pandas, Matplotlib, Seaborn  
- (Optional) Jupyter Notebook or Google Colab for interactive exploration

Install the required packages:
```bash
pip install torch numpy pandas matplotlib seaborn
```

---

## How to Run

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/YourUsername/DeepLearningXOR_MNIST.git
   ```
2. **Navigate** into the repository folder:
   ```bash
   cd DeepLearningXOR_MNIST
   ```
3. **Open** each `.ipynb` file in Jupyter Lab/Notebook or Google Colab:
   ```bash
   jupyter notebook
   ```
   - _PyTorch XOR Implementation and Analysis.ipynb_
   - _Hyperparameter Exploration for XOR.ipynb_
   - _Deep Learning on MNIST: From Simple Networks to CNN Architectures.ipynb_

4. **Run All Cells** in each notebook to reproduce the experiments and see outputs (plots, metrics, logs).

---

## Notable Highlights

- **Bypass Connections for XOR**: Showcases the value of adding direct input-to-output links in very small networks.  
- **Comprehensive Hyperparameter Search**: Uses manual and parallel runs to find optimal learning rates and hidden units for XOR.  
- **CNN Filters Visualization**: Demonstrates how convolution kernels learn digit edges and strokes in MNIST classification.  
- **Evaluation Metrics**: Logs standard metrics (loss, precision, recall, F1, accuracy) and includes confusion matrices for detailed model assessments.

---

## Results and Observations

- For XOR, even a **1-neuron hidden layer** can succeed with a properly tuned learning rate and bypass.  
- Larger hidden layers converge but may require more epochs and advanced optimizers.  
- On MNIST, straightforward CNNs **quickly surpass 99% accuracy**, and dropout layers help generalize.  
- Visualizing filters offers insights into how CNNs capture **edges and shapes** crucial to digit recognition.

---

## Contributing

Feel free to submit issues, fork the repo, or open pull requests for improvements and new experiments. All contributions to enhance model architectures, logging, or data exploration are welcome.

---

## License

This project is available under the [MIT License](LICENSE).  
Use it freely for personal or commercial purposes, but please retain the original license and references to the contributors.

---

**Enjoy experimenting with XOR and MNIST!**  
If you have any questions or suggestions, please create an issue or reach out directly via GitHub.
```
