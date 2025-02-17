# Deep Learning Notebooks for XOR, MNIST & LSTM Language Modeling

This repository contains several Jupyter notebooks demonstrating different deep learning techniques using PyTorch and TensorFlow. The notebooks cover classical problems (XOR and MNIST) as well as advanced language modeling with LSTM networks.

---

## Repository Overview

1. **PyTorch XOR Implementation and Analysis**  
   - Implements a simple neural network to solve the classic XOR problem.  
   - Demonstrates network design with bypass connections and minimal neuron architectures to achieve zero training loss.

2. **Hyperparameter Exploration for XOR**  
   - Extends the XOR model with experiments on multiple hyperparameters (learning rates, hidden units, bypass connections).  
   - Tracks convergence speed and stability using early stopping and performance logs.

3. **Deep Learning on MNIST: From Simple Networks to CNNs**
   ![image](https://github.com/user-attachments/assets/258d15b3-a354-47f3-8155-3fb8974d7ff1)

   - Demonstrates both fully connected and convolutional neural networks for digit classification on MNIST.  
   - Achieves over 99% validation accuracy and visualizes learned filters to reveal spatial patterns.
![Uploading image.png…]()

4. **LSTM Language Model Integration with Forward/Reverse Training**  
![image](https://github.com/user-attachments/assets/39cd1390-9731-4a50-b275-8baeac86a796)

   - **Notebook:** [LSTM_Model_Perplexity.ipynb](https://github.com/ShovalBenjer/deep_learning_neural_networks/blob/main/LSTM_Model_Perplexity.ipynb)  
   - **Highlights:**
     - **Data Splitting:** The dataset is divided into training (80%), validation (10%), and test (10%) sets.
     - **Perplexity Computation:** The notebook computes perplexity on each split to gauge model performance.
     - **Bidirectional Training:** Supports training the LSTM both in forward (natural) order and in reverse order using Keras’s `go_backwards` flag.
     - **Model Variations:** Trains four models – one-layer vs. two-layer LSTM, each in both forward and reverse modes.
     - **Sentence Generation & Interactive UI:** Provides functions to generate sentences (with adjustable temperature) and an interactive UI to predict subsequent words from a seed.
     - **Logging:** Training progress and metrics are logged via TensorBoardX.
     - **Additional Evaluation:** Computes the probability for custom sentences (e.g., "love i cupcakes") and logs perplexity across all splits.

---

```markdown

## Requirements

- Python 3.x  
- [PyTorch](https://pytorch.org/) for the XOR/MNIST notebooks  
- [TensorFlow](https://www.tensorflow.org/) & [TensorBoardX](https://tensorboardx.readthedocs.io/) for the LSTM notebook  
- NumPy, Pandas, Matplotlib, Seaborn  
- Jupyter Notebook or Google Colab for interactive exploration

Install the required packages:

```bash
pip install torch numpy pandas matplotlib seaborn tensorflow tensorboardX
```

---

## How to Run

1. **Clone or Download** the repository:
   ```bash
   git clone https://github.com/YourUsername/DeepLearningNotebooks.git
   ```
2. **Navigate** into the repository folder:
   ```bash
   cd DeepLearningNotebooks
   ```
3. **Open the Notebooks** in your preferred environment (Jupyter Notebook, Jupyter Lab, or Google Colab):
   - _PyTorch XOR Implementation and Analysis.ipynb_
   - _Hyperparameter Exploration for XOR.ipynb_
   - _Deep Learning on MNIST: From Simple Networks to CNN Architectures.ipynb_
   - _LSTM_Model_Perplexity.ipynb_

4. **Run All Cells** in each notebook to reproduce experiments, observe outputs (plots, metrics, logs), and explore interactive functionalities (especially in the LSTM notebook).

---

## Notable Highlights

- **XOR & MNIST Experiments:**
  - Demonstrates the impact of bypass connections in minimal networks.
  - Extensive hyperparameter search to optimize model performance.
  - Visualization of CNN filters to understand feature extraction in MNIST classification.

- **LSTM Language Model:**
  - Integrates both forward and backward training to enhance language model robustness.
  - Uses perplexity as a metric to quantify model predictive performance.
  - Provides sentence generation at different temperatures, enabling exploration of creative text outputs.
  - Interactive UI function allows users to input a seed word and predict the next word.

---

## Results and Observations

- **XOR Models:**  
  Even a minimal network (with a 1-neuron hidden layer and bypass connections) can achieve near-zero training loss with proper hyperparameter tuning.

- **MNIST Networks:**  
  Convolutional architectures quickly surpass 99% accuracy, with dropout layers contributing to better generalization.

- **LSTM Models:**  
  - Training both forward and backward models reveals differences in perplexity across the train, validation, and test splits.
  - The dual approach (one-layer vs. two-layer) provides insights into model capacity and the effect of network depth on language modeling.
  - Generated sentences vary with temperature, illustrating the balance between randomness and determinism in text generation.

---

## Contributing

Contributions are welcome! Feel free to fork the repository, submit issues, or open pull requests to improve model architectures, experiment setups, or documentation. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines.

---

## License

This project is available under the [MIT License](LICENSE). Use it freely for personal or commercial purposes while retaining the original license and contributor credits.

---

## Contact

For questions, feedback, or collaboration opportunities, please contact:  

**Shoval Benjer**  
Creative Data Scientist | Tel Aviv - Jaffa, ISR  
GitHub: [ShovalBenjer](https://github.com/ShovalBenjer)  
Email: shovalb9@gmail.com  

---

**Enjoy exploring and experimenting with these deep learning techniques!**
