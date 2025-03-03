# Deep Learning Course Guidebook

## About this Study Guide

This guidebook serves as a comprehensive resource for understanding neural networks and machine learning, designed to complement advanced deep learning coursework. It consolidates lecture materials, quizzes, essays, and glossaries into a structured format for effective learning and review. The content combines clear explanations with mathematical notations and equations, aiming to provide a robust foundation in key deep learning concepts.

This guide is organized to cover fundamental to advanced topics, starting with perceptrons and building up to complex architectures like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers. It aims to be a valuable reference for anyone looking to deepen their understanding of neural networks and machine learning.

---

## Table of Contents

**I. Foundational Concepts**
1.  **[Perceptrons](#lecture-2-perceptron-study-guide)**
    - [Perceptron](#perceptrons)
    - [Binary Threshold Unit (BTU)](#types-of-neurons)
    - [Perceptron Learning Algorithm](#perceptron-learning-algorithm)
    - [Linear Separation](#linear-separation)
    - [Limitations of Linear Separation](#limitations-of-linear-separation)
2.  **[Building Artificial Neural Networks](#lecture-1-artificial-neural-networks-and-deep-learning)**
    - [Building Artificial Neural Networks](#building-artificial-neural-networks)
    - [Visible Units](#building-artificial-neural-networks)
    - [Deep Networks](#building-artificial-neural-networks)
    - [Shallow Networks](#building-artificial-neural-networks)
3.  **[Types of Neurons](#types-of-neurons)**
    - [Types of Neurons](#types-of-neurons)
    - [Binary Threshold Unit (BTU)](#types-of-neurons)
    - [Logistic Neurons (Sigmoid)](#types-of-neurons)
    - [Linear Unit](#types-of-neurons)
    - [Rectified Linear Unit (ReLU)](#types-of-neurons)
    - [Hyperbolic Tangent (Tanh)](#types-of-neurons)
    - [Leaky ReLU](#types-of-neurons)
    - [Stochastic Binary Neurons](#types-of-neurons)
4.  **[Types of Machine Learning](#types-of-machine-learning)**
    - [Types of Machine Learning](#types-of-machine-learning)
    - [Supervised Learning](#types-of-machine-learning)
    - [Unsupervised Learning](#types-of-machine-learning)
5.  **[Linear and Logistic Regression](#linear-and-logistic-regression)**
    - [Linear and Logistic Regression](#linear-and-logistic-regression)
    - [Linear Regression](#linear-and-logistic-regression)
    - [Logistic Regression](#linear-and-logistic-regression)
6.  **[Gradient Descent and SGD](#gradient-descent-and-sgd)**
    - [Gradient Descent and SGD](#gradient-descent-and-sgd)
    - [Gradient Descent (GD)](#gradient-descent-and-sgd)
    - [Stochastic Gradient Descent (SGD)](#gradient-descent-and-sgd)
    - [Mini-Batch SGD](#gradient-descent-and-sgd)
7.  **[Error Backpropagation](#lecture-5-error-backpropagation)**
    - [Backpropagation](#backpropagation)
    - [Backpropagation Algorithm](#backpropagation)
    - [Delta Values](#backpropagation)
    - [Non-Linear Activation Functions](#backpropagation)
    - [Vanishing Gradients](#backpropagation)
    - [Weight Initialization](#backpropagation)
8.  **[Cost Functions](#cost-functions)**
    - [Cost Functions](#cost-functions)
    - [Cross-Entropy (CE)](#cost-functions)
    - [Mean Squared Error (MSE)](#cost-functions)
9.  **[Overfitting and Regularization](#overfitting-and-regularization)**
    - [Overfitting and Regularization](#overfitting-and-regularization)
    - [Overfitting](#overfitting-and-regularization)
    - [Regularization](#overfitting-and-regularization)
    - [Regularization Techniques](#overfitting-and-regularization)
    - [Weight Decay](#overfitting-and-regularization)
    - [Dropout Regularisation](#overfitting-and-regularization)
    - [Early Stopping](#overfitting-and-regularization)
    - [Data Augmentation & Noise Injection](#overfitting-and-regularization)
10. **[Deep Architectures](#deep-architectures)**
    - [Deep Architectures](#deep-architectures)
    - [Autoencoders](#deep-architectures)
    - [Bottleneck Layer](#deep-architectures)
    - [Layer-wise Training](#deep-architectures)
11. **[Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)**
    - [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
    - [Convolution Layer](#convolutional-neural-networks-cnns)
    - [Pooling Layer](#convolutional-neural-networks-cnns)
    - [Parameter Sharing](#convolutional-neural-networks-cnns)
    - [Receptive Field](#convolutional-neural-networks-cnns)
    - [Translation Invariance](#convolutional-neural-networks-cnns)
12. **[Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rns)**
    - [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rns)
    - [RNN Fundamentals](#recurrent-neural-networks-rns)
    - [Hidden State](#recurrent-neural-networks-rns)
    - [Bidirectional RNN](#recurrent-neural-networks-rns)
    - [Backpropagation Through Time (BPTT)](#recurrent-neural-networks-rns)
    - [LSTM and GRU](#lstm-and-gru)
    - [Long Short-Term Memory (LSTM)](#lstm-and-gru)
    - [Gated Recurrent Unit (GRU)](#lstm-and-gru)
13. **[Attention Mechanisms and Transformers](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)**
    - [Attention Mechanisms](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
    - [Attention Mechanism Principle](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
    - [Query-Key-Value Design](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
    - [Self-Attention](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
    - [Transformer Architecture](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
    - [Positional Encoding](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
    - [Beam Search](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
    - [BERT and XLNet](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
14. [Lecture 1: Artificial Neural Networks and Deep Learning](#lecture-1-artificial-neural-networks-and-deep-learning)
15. [Lecture 2: Perceptron Study Guide](#lecture-2-perceptron-study-guide)
16. [Lecture 3: Perceptron Learning](#lecture-3-perceptron-learning)
17. [Lecture 4: Neural Networks and Gradient Descent](#lecture-4-neural-networks-and-gradient-descent)
18. [Lecture 5: Error Backpropagation](#lecture-5-error-backpropagation)
19. [Lecture 6: Convolutional Neural Networks](#lecture-6-convolutional-neural-networks)
20. [Lecture 7: Recurrent Neural Networks](#lecture-7-recurrent-neural-networks)
21. [Lecture 8: Neural Networks and Backpropagation](#lecture-8-neural-networks-and-backpropagation)
22. [Lecture 9: Attention Mechanisms, Transformers, and Advanced Topics](#lecture-9-attention-mechanisms-transformers-and-advanced-topics)
23. [Additional Resources & Final Thoughts](#additional-resources--final-thoughts)

**IV. Project Specifics**
24. [About Project itself](#about-project-itself)
25. [Requirements](#requirements)
26. [How to Run](#how-to-run)
27. [Notable Highlights](#notable-highlights)
28. [Results and Observations](#results-and-observations)
29. [Contributing](#contributing)
30. [License](#license)
31. [Contact](#contact)

---

## Perceptrons

In artificial neural networks, **weights** ($w_{ij}$) are used to indicate the strength of a synapse. A positive weight indicates an excitatory connection, while a negative weight indicates an inhibitory connection. The influence of one neuron on another is controlled by the "strength" of the synapse (positive or negative). The synaptic weights are adjusted during learning, allowing the network to perform useful computations such as object recognition, language understanding, planning, and body control.

### Mathematical Representation:
- **Weighted Sum (Z):**
  $$z_i = \sum_{j} w_{ij} x_j + b$$
  where $w_{ij}$ are the weights, $x_j$ are the inputs, and $b$ is the bias.
- **Activation Function (g):**
  $$y_i = g(z_i)$$
  where $g$ is the activation function.
  Figure 1: Artificial Neuron

## Building Artificial Neural Networks

### Building Artificial Neural Networks
- **Visible Units**: Neurons in the input and output layers are called visible units.
- **Deep Networks**: Networks with more than one hidden layer are called deep networks.
- **Shallow Networks**: Networks without hidden layers are computationally limited.

## Types of Neurons

### Types of Neurons
Figure 2: Neuron Types Overview

### Binary Threshold Unit (BTU)
$$y_i = \sigma(z_i) =
\begin{cases}
1, & \text{if } z_i > 0 \\
0, & \text{if } z_i \leq 0
\end{cases}$$
The input to this activation function is between 0 and 1.
Figure 3: Binary Threshold Unit Activation Function

### Logistic Neurons (Sigmoid)
$$y_i = g(z_i) = \frac{1}{1 + e^{-z_i}}$$
The output is a real number between 0 and 1. When $z = 0$, the output is 0.5. For negative $z$, the output is below 0.5, and for positive $z$, the output is above 0.5.
Figure 4: Sigmoid Activation Function

### Linear Unit
$$y_i = z_i$$
The output is a linear function of the weighted sum of inputs.

### Rectified Linear Unit (ReLU)
$$y_i = \max(0, z_i)$$
The output is $z_i$ if positive, otherwise 0.
Figure 5: ReLU Activation Function

### Hyperbolic Tangent (Tanh)
$$y_i = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$
The output is between -1 and 1.
Figure 6: Tanh Activation Function

### Leaky ReLU
$$g(z) = \max(z, \alpha z)$$
where $\alpha$ is a fraction between 0 and 1. If $\alpha = 0$, it becomes a standard ReLU.

### Stochastic Binary Neurons
Stochastic Binary Neurons: Belief Network. In Stochastic Binary Neurons, the output is probabilistic. The 'belief' of the network is between 0 and 1 - the output is probabilistic.
$$z = \sum_{j} w_{ji} x_j + W_0, \quad P(s_i=1) = \frac{1}{1+\exp(-z_i)}$$
$y \rightarrow 1$ if $P(s_i = 1) \rightarrow 1$
$y \rightarrow 0$ if $P(s_i = 1) \rightarrow 0$
Figure 7: Stochastic Binary Neuron Activation Function

## Linear Separation

### Linear Separation
In the input space, each input example is a point. A hyperplane defined by a vector of $n+1$ weights separates the space into positive and negative regions based on the sign of the dot product $WX$.

### Example:
For a 2D input, the hyperplane is a straight line:
$$w_1 x_1 + w_2 x_2 + b > 0 \quad \text{(Positive)}$$
$$w_1 x_1 + w_2 x_2 + b \leq 0 \quad \text{(Negative)}$$

## Limitations of Linear Separation

### Limitations of Linear Separation
Linear separation has limitations, such as the inability to separate XOR. A single perceptron without hidden layers cannot solve problems like XOR or parity.

## Perceptron Learning Algorithm

### Perceptron Learning Algorithm
The goal is to find weights $W$ that define a separating hyperplane:
$$w_1 x_1 + w_2 x_2 + \dots + w_n x_n + w_0 = 0$$

### Steps:
1. Initialize weights randomly.
2. For each training example, compute the output:
   $$y_k = \sigma\left(\sum_{j} w_j x_{kj}\right)$$
3. Compute the error:
   $$\text{Error} = t_k - y_k$$
4. Update the weights:
   $$\Delta w_i = \lambda (t_k - y_k) x_{ki}$$
   $$w_i = w_i + \Delta w_i$$
   where $\lambda$ is the learning rate.

## Types of Machine Learning

### Types of Machine Learning
Figure 8: Types of Machine Learning Overview

### Supervised Learning
- **Training**: Learn from labeled examples to minimize errors.
- **Validation**: Evaluate the model on unseen data.
- **Testing**: Use the model in real-world applications.

### Unsupervised Learning
- **Goal**: Find patterns or structure in unlabeled data.

## Linear and Logistic Regression

### Linear and Logistic Regression

### Linear Regression
$$y = w_1 x + w_0$$
The goal is to find parameters $w_0$ and $w_1$ that minimize the mean squared error (MSE).

### Logistic Regression
$$y = \frac{1}{1 + e^{-(w_1 x + w_0)}}$$
Used for binary classification.

## Gradient Descent and SGD

### Gradient Descent and SGD

### Gradient Descent (GD)
- Updates weights after computing the average gradient over all training examples.
- **Batch Update**:
  $$\Delta w_i = -\lambda \frac{\partial \text{loss}}{\partial w_i}$$

### Stochastic Gradient Descent (SGD)
- Updates weights after each training example.
- **Mini-Batch SGD**: Updates weights after a small batch of examples.

## Backpropagation

### Backpropagation
Backpropagation is used to compute gradients for updating weights in neural networks.

### Backpropagation Algorithm
- **Steps:**
  1. Initialize weights randomly.
  2. Forward pass: Compute activations for each layer.
  3. Backward pass: Compute deltas ($\delta$) for each layer.
  4. Update weights using the computed gradients.

### Delta Values
Delta ($\delta$): The error signal for a neuron used in weight updates. Mathematically, for the output layer ($L$):
$$ \delta_i^{(L)} = \frac{\partial Loss}{\partial z_i^{(L)}} $$
And for hidden layers ($l < L$):
$$ \delta_i^{(l)} = \frac{\partial Loss}{\partial z_i^{(l)}} = \frac{\partial a_i^{(l)}}{\partial z_i^{(l)}} \sum_{j} W_{ji}^{(l+1)} \delta_j^{(l+1)} $$

### Non-Linear Activation Functions
Non-linear activation functions are essential because they enable networks to model complex relationships. Without non-linear activations, multiple layers would collapse into a single linear transformation, limiting the network's capacity.

### Vanishing Gradients
Vanishing Gradient: A phenomenon where gradients diminish as they are propagated through many layers, slowing learning in early layers. This is particularly pronounced in deep networks and with activation functions like Sigmoid and Tanh when their input leads to saturation.

### Weight Initialization
Weight Initialization: Techniques (like Xavier or He initialization) used to set initial weights for stable training. Proper initialization helps prevent vanishing or exploding gradients and ensures that the network starts learning effectively. Small random weights are typically preferred to break symmetry and avoid saturation at the start of training.

## Cost Functions

### Cost Functions

### Cross-Entropy (CE)
$$\text{CE}(y, t) = -t \log(y) - (1 - t) \log(1 - y)$$
Cross-Entropy is commonly used for classification tasks, especially with Softmax output layers.

### Mean Squared Error (MSE)
$$\text{MSE}(t, y) = (t - y)^2$$
Mean Squared Error is typically used for regression tasks where the output is a continuous value.

## Overfitting and Regularization

### Overfitting and Regularization

### Overfitting
- Occurs when the model learns noise in the training data, performing well on the training set but poorly on unseen data.
- **Solutions**:
  - Add more data: Increasing the dataset size helps the model generalize better by reducing the influence of noise specific to the training set.
  - Use regularization (e.g., L2 regularization, Dropout): Regularization techniques constrain the model's complexity, preventing it from fitting noise.
  - Early stopping: Monitor the validation error and stop training when it starts to increase, preventing overfitting to the training data.

### Regularization
- Penalizes large weights to prefer simpler models, improving generalization.

### Regularization Techniques
L2 Regularization (Weight Decay): Adds a penalty term to the loss function proportional to the square of the weights.
$$\text{RegLoss}(w) = \text{Loss}(w) + \frac{\gamma}{2} \sum_{j=1}^{n} w_j^2$$
where $\gamma$ is the regularization strength.

### Weight Decay
Weight Decay (L2 Regularization): Penalizes large weights, encouraging simpler models and preventing overfitting. It is mathematically equivalent to weight decay, where weights are slightly reduced in each update step.

### Dropout Regularisation
Dropout Regularisation: Randomly deactivates neurons during training. This prevents neurons from co-adapting and forces the network to learn more robust features. During training, neurons are dropped out with a probability $p$, and during testing, all neurons are used.

### Early Stopping
Early Stopping: Monitors validation performance and stops training when validation loss starts to increase, preventing overfitting by halting training at the point of best generalization.

### Data Augmentation & Noise Injection
Data Augmentation & Noise Injection: Techniques to improve generalization. Data augmentation increases the diversity of the training set by applying transformations (e.g., rotations, flips) to input data. Noise injection adds random noise to inputs or intermediate layers, making the model more robust to input variations and acting as a regularizer.

## Deep Architectures

### Deep Architectures
Deep networks learn hierarchical features, starting from simple patterns in early layers to complex patterns in deeper layers. Depth enables the network to learn complex functions more efficiently than shallow networks.

### Autoencoders
- Unsupervised learning method that compresses data to find useful features. Autoencoders learn efficient data codings in an unsupervised manner by training the network to ignore signal-noise.
- **Bottleneck Layer**: Forces the network to learn a compressed representation. The bottleneck layer in autoencoders enforces a compressed representation of the input data, capturing the most salient features in a lower-dimensional space.

### Layer-wise Training
Layer-wise Training (for Deep Autoencoders): A method to train deep autoencoders layer by layer to overcome optimization difficulties and vanishing gradients. Each layer is pre-trained independently, often using techniques like Restricted Boltzmann Machines (RBMs) or denoising autoencoders, before stacking and fine-tuning the entire network.

## Convolutional Neural Networks (CNNs)

### Convolutional Neural Networks (CNNs)
CNNs use local connectivity and shared weights to detect features in images efficiently. They are particularly effective for image recognition and other grid-like data.

### Convolution Layer
- Applies filters (kernels) to local regions of the input to extract features.
- **Output Size**: The size of the output feature map from a convolutional layer is determined by the input size ($W_1$), filter size ($F$), padding ($P$), and stride ($S$):
  $$W_2 = \frac{(W_1 - F + 2P)}{S} + 1$$
  where $W_1$ is the input width, $F$ is the filter width, $P$ is padding, and $S$ is stride.

### Pooling Layer
- Reduces the spatial dimensions of the feature maps, reducing computation and making feature detection more robust to location variations.
- **Max Pooling**: Takes the maximum value in each window, emphasizing the most prominent features.
- **Average Pooling**: Takes the average value in each window, smoothing features and reducing noise.

### Parameter Sharing
Parameter Sharing in CNNs significantly reduces the number of model parameters. By using the same filter weights across different locations in the input, CNNs achieve translation invariance and learn features that are useful regardless of their spatial position.

### Receptive Field
Receptive Field: The region in the input space that affects a particular neuron’s output. In CNNs, deeper layers have larger receptive fields, allowing them to capture more complex and global features.

### Translation Invariance
Translation Invariance: CNNs achieve translation invariance through parameter sharing and pooling, making them robust to translations of features in the input images. This means that the network can detect a feature regardless of where it appears in the image.

## Recurrent Neural Networks (RNNs)

### Recurrent Neural Networks (RNNs)
RNNs are designed to handle sequential data by maintaining a hidden state that captures information from previous time steps. They are well-suited for tasks like natural language processing, speech recognition, and time series analysis.

### RNN Fundamentals
RNNs process sequential data by iteratively updating a hidden state. At each time step $t$, the hidden state $s_t$ is updated based on the current input $x_t$ and the previous hidden state $s_{t-1}$.

### Hidden State
- **Hidden State Update Rule**: The hidden state $s_t$ at time step $t$ is computed as:
  $$s_t = f(U x_t + W s_{t-1})$$
  where $U$ is the weight matrix for the input $x_t$, $W$ is the weight matrix for the previous hidden state $s_{t-1}$, and $f$ is an activation function (e.g., tanh or ReLU).

### Bidirectional RNN
- Processes the input sequence in both forward and backward directions. Bidirectional RNNs combine two RNNs, one processing the sequence from beginning to end and the other from end to beginning. This allows the model to capture contextual information from both past and future time steps, enhancing performance in tasks like sequence labeling and machine translation.

### Backpropagation Through Time (BPTT)
Backpropagation Through Time (BPTT): A method for training RNNs. BPTT extends the standard backpropagation algorithm to handle sequential data by "unrolling" the RNN over time. In BPTT, the RNN is treated as a deep feedforward network where each time step corresponds to a layer. Gradients are then computed through this unrolled network and accumulated over time.

### LSTM and GRU

### Long Short-Term Memory (LSTM)
- **Long Short-Term Memory (LSTM)**: A type of RNN architecture designed to mitigate the vanishing gradient problem and effectively learn long-range dependencies in sequential data. LSTMs introduce a cell state and gating mechanisms to control the flow of information over time.
  - **Cell State**: Maintains long-term memory across time steps, acting as a conveyor belt that carries relevant information through the sequence.
  - **Gates**: Control the flow of information into and out of the cell state. LSTMs have three main gates:
      - **Forget Gate ($f_t$)**: Decides what information to discard from the cell state.
      - **Input Gate ($i_t$)**: Decides what new information to store in the cell state.
      - **Output Gate ($o_t$)**: Controls what information from the cell state to output.
Figure 9: LSTM Architecture

### Gated Recurrent Unit (GRU)
- **Gated Recurrent Unit (GRU)**: A simplified version of LSTM that combines the forget and input gates into a single "update gate" and merges the cell state and hidden state. GRUs are computationally more efficient and often perform comparably to LSTMs in many tasks.
Figure 10: GRU Architecture

## Attention Mechanisms and Transformers

### Attention Mechanisms and Transformers

### Attention Mechanism Principle
Attention Mechanism Principle: Allows the model to weigh different parts of the input differently when producing output. This enables the model to focus on the most relevant parts of the input sequence, significantly improving performance, especially in tasks like machine translation and image captioning, where long-range dependencies and input focus are crucial.

### Query-Key-Value Design
Query-Key-Value Design: Attention mechanisms typically use a Query-Key-Value framework. The query, key, and value are vectors derived from the input. The attention mechanism computes a compatibility score between the query and each key to determine the relevance of each value. These scores are then used to weigh the values, producing a context-aware representation.

### Self-Attention
Self-Attention (Intra-Attention): Allows the model to relate different positions of a single input sequence to compute a representation of the same sequence. In self-attention, the queries, keys, and values are all derived from the same input sequence. This mechanism is particularly effective in capturing long-range dependencies within a sequence without relying on recurrent structures.

### Transformer Architecture
Transformer Architecture: A neural network architecture based entirely on attention mechanisms, notably self-attention. Transformers replace RNNs with multi-head attention layers, feedforward networks, and residual connections. This architecture enables parallel processing of the input sequence, leading to faster training and better scalability, especially for long sequences. Transformers have achieved state-of-the-art results in various NLP tasks, including machine translation, text summarization, and language modeling.

### Positional Encoding
Positional Encoding: Since Transformers process inputs in parallel and do not inherently account for the order of sequence elements (unlike RNNs), positional encoding is added to the input embeddings. Positional encodings provide information about the position of each token in the sequence, enabling the model to utilize sequence order. Common positional encoding methods include sinusoidal functions.

### Beam Search
Beam Search: A heuristic search strategy used in sequence generation tasks, particularly in models like Transformers for machine translation and text generation. Beam search explores a beam of top-k candidate sequences at each decoding step, keeping track of multiple promising hypotheses to find a high-probability output sequence.

### BERT and XLNet
BERT (Bidirectional Encoder Representations from Transformers) and XLNet: Large pre-trained Transformer models that have revolutionized Natural Language Processing. BERT uses a masked language modeling objective and next sentence prediction, while XLNet employs a permutation language modeling approach. These models learn deep bidirectional representations and achieve state-of-the-art performance on a wide range of NLP tasks after fine-tuning.

## Lecture 1: Artificial Neural Networks and Deep Learning

**3Blue1Brown Video:**  
[Neural Networks – The Math of Intelligence](https://www.youtube.com/watch?v=3v7NPR0A9I4)

### Overview & Key Topics
- **Primary Goals:** Understand the fundamentals of AI powered by ANNs, explore deep learning architectures, practice using PyTorch, and complement traditional ML courses.
- **Neuron Structure:** An artificial neuron computes a weighted sum of inputs, adds a bias, and passes this through an activation function (e.g., sigmoid, ReLU, tanh).
- **Hebb's Rule:** “Fire together, wire together” – the idea that simultaneous activation strengthens connections.
- **AI Spectrum:** Definitions of Weak AI (narrow task-specific), General AI (multi-task learning), and Strong AI (human-like cognition).
- **Backpropagation:** The method used to update weights by computing gradients of the loss function.
- **Synaptic Effects:** Difference between excitatory (positive weights) and inhibitory (negative weights) synapses.
- **Datasets & Applications:** Overview of ImageNet’s role in computer vision; introduction to GANs for generative modeling.

### Glossary
- **Activation Function:** Applies non-linearity (e.g., sigmoid outputs between 0 and 1).
- **Backpropagation:** Algorithm for updating weights via gradients.
- **Hebb's Rule:** “Fire together, wire together” – strengthening of connections through simultaneous activation.
- **ImageNet:** A benchmark dataset with millions of labeled images.
- **GAN (Generative Adversarial Network):** Comprises a generator and discriminator competing adversarially.
- **ReLU:** Outputs the input if positive, otherwise zero.
- **Tanh:** Outputs values between -1 and 1.
- **PyTorch:** A deep learning framework for building neural networks.
- **Weights:** In artificial neural networks, **weights** are used to indicate the strength of a synapse. Synaptic weights are adjusted during learning, allowing the network to perform useful computations.

---

## Lecture 2: Perceptron Study Guide

**3Blue1Brown Video:**  
[Perceptron and its Limitations](https://www.youtube.com/watch?v=mCqTJd3N7qI)

### Overview & Key Topics
- **Perceptron Basics:** Understand the Binary Threshold Unit (BTU) – its weighted sum, bias, and thresholding.
- **Logic Gates Implementation:** How perceptrons can model AND, NOT, but not XOR.
- **Building Theorem:** States that any Boolean function can be implemented by a perceptron with a hidden layer.
- **Linear Separability:** The concept that a dataset must be linearly separable for a single-layer perceptron to work.
- **Geometric Interpretation:** A perceptron defines a hyperplane that separates classes.

### Glossary
- **Perceptron:** A basic neural network unit for binary classification.
- **Binary Threshold Unit (BTU):** The activation that outputs 1 if the weighted sum exceeds a threshold.
- **Linear Separability:** The ability to divide classes using a straight line or hyperplane.
- **Building Theorem:** The concept that multilayer perceptrons can implement any Boolean function.
- **Hidden Layer:** An intermediate layer in a multilayer perceptron that enables non-linear decision boundaries.
- **Hyperplane:** A flat subspace of dimension $n-1$ in an $n$-dimensional space, used by perceptrons for linear separation.

---

## Lecture 3: Perceptron Learning

**3Blue1Brown Video:**  
[Understanding Perceptron Learning](https://www.youtube.com/watch?v=05N2t0tJ1jI)

### Overview & Key Topics
- **Perceptron Learning Algorithm:** An iterative method to adjust weights based on classification errors.
- **Error Calculation & Weight Update:** Update rule: \(\Delta w = \lambda (t - y)x\).
- **Learning Rate (λ):** Controls the adjustment magnitude.
- **Information Capacity:** Relationship between the number of inputs and the number of patterns the perceptron can learn.
- **Visualization:** Visualizing weights as an image to interpret learned features (e.g., in image recognition).

### Glossary
- **Learning Rate (λ):** A hyperparameter that dictates the size of weight updates.
- **Epoch:** One full pass through the training dataset.
- **Hyperplane:** The decision boundary in the feature space.
- **Cone of Solutions:** The set of weight vectors that yield correct classifications.
- **Information Capacity:** The ratio of patterns learned to inputs available; beyond a threshold, learning fails.

---

## Lecture 4: Neural Networks and Gradient Descent

**3Blue1Brown Video:**  
[Gradient Descent – The Math of Optimization](https://www.youtube.com/watch?v=0Lt9w-BxKFQ)

### Overview & Key Topics
- **Gradient Descent (GD):** The iterative algorithm to minimize a loss function by moving in the direction of the steepest descent.
- **Stochastic and Mini-Batch GD:** How using subsets of data (mini-batches) introduces stochasticity and speeds up learning.
- **Learning Rate:** Its critical role in determining convergence speed and stability.
- **Chain Rule:** Application of the chain rule to compute gradients for deep networks (backpropagation).
- **Loss Functions:** Explanation of MSE for regression and Cross-Entropy for classification, along with the SoftMax function for multi-class problems.
- **One-Hot Encoding:** How categorical labels are represented for training with Cross-Entropy loss.

### Glossary
- **Gradient Descent (GD):** An optimization algorithm that updates parameters by moving in the direction of the negative gradient.
- **Stochastic Gradient Descent (SGD):** GD performed on a single example (or small batch), adding noise but speeding up computation.
- **Chain Rule:** A calculus rule that allows the derivative of a composite function to be computed as the product of derivatives.
- **SoftMax Function:** Converts a vector of logits into a probability distribution over classes.
- **Cross-Entropy Loss:** Measures the discrepancy between the predicted probability distribution and the true distribution.
- **One-Hot Encoding:** Represents categorical variables as binary vectors.

---

## Lecture 5: Error Backpropagation

**3Blue1Brown Video:**  
[Backpropagation Explained](https://www.youtube.com/watch?v=2vI82f-0qxI)

### Overview & Key Topics
- **Purpose of Non-Linear Activation Functions:** They allow neural networks to learn complex, non-linear relationships.
- **Loss Function Role:** Measures prediction error; common examples are MSE and Cross-Entropy.
- **MLP Structure:** Multi-Layer Perceptron composed of input, hidden, and output layers.
- **Chain Rule in Backpropagation:** How gradients are computed layer-by-layer.
- **Normalisation:** Its importance in preventing saturation and ensuring balanced learning.
- **Vanishing Gradients:** The challenge in deep networks; strategies like careful weight initialization help.
- **Stochastic Gradient Descent:** Its use for updating weights with mini-batches.
- **Computational Graphs:** Tools for visualizing and understanding gradient flow.
- **Delta Values:** Computation differences between output and hidden layers during backpropagation.
- **Weight Initialization:** Techniques to prevent vanishing gradients.

### Glossary
- **Backpropagation:** The method of calculating gradients in neural networks by propagating errors backward.
- **Delta (𝛿):** The error signal for a neuron used in weight updates.
- **Computational Graph:** A graph representing the operations in a neural network, used for gradient computation.
- **Vanishing Gradient:** A phenomenon where gradients diminish as they are propagated through many layers, slowing learning in early layers.
- **Weight Initialization:** Techniques (like Xavier or He initialization) used to set initial weights for stable training.

---

## Lecture 6: Convolutional Neural Networks

**3Blue1Brown Video:**  
[Convolutional Neural Networks Explained](https://www.youtube.com/watch?v=7VeUPuFGJHk)

### Overview & Key Topics
- **CNN Fundamentals:** Convolution operations, filters/kernels, feature maps, and pooling layers.
- **Parameter Sharing:** How convolutional layers reduce the number of parameters compared to fully connected layers.
- **Receptive Field:** The region of the input each neuron “sees,” which grows with network depth.
- **Pooling:** Downsampling methods (e.g., max pooling) that reduce spatial dimensions and help with invariance.
- **Advanced Architectures:** Overview of residual networks (ResNets) and Inception modules.
- **Data Augmentation & Transfer Learning:** Techniques to improve model robustness and leverage pre-trained networks (e.g., on ImageNet).

### Glossary
- **Convolution:** A mathematical operation for feature extraction by sliding a filter over input data.
- **Filter/Kernel:** A set of learnable weights in a CNN that extracts features from the input.
- **Pooling:** A downsampling operation that reduces spatial dimensions.
- **Receptive Field:** The region in the input space that affects a particular neuron’s output.
- **Residual Network (ResNet):** A CNN with skip connections that help with training very deep networks.
- **Inception Module:** A block that applies multiple convolutions of different sizes in parallel.

---

## Lecture 7: Recurrent Neural Networks

**3Blue1Brown Video:**  
[Recurrent Neural Networks Explained](https://www.youtube.com/watch?v=0Lt9w-BxKFQ)

### Overview & Key Topics
- **RNN Fundamentals:** How RNNs process sequences using hidden states that capture temporal context.
- **Unfolding RNNs:** Visualizing the RNN over time to apply backpropagation through time (BPTT).
- **Long Short-Term Memory (LSTM) & GRU:** RNN variants designed to overcome vanishing gradients with gating mechanisms.
- **Sequence-to-Sequence Models:** Encoder-decoder architectures for tasks like machine translation.
- **Attention Mechanisms:** Brief introduction to how attention (covered in detail in Lecture 5) augments RNNs.

### Glossary
- **RNN:** A neural network that processes sequential data by maintaining and updating a hidden state.
- **Backpropagation Through Time (BPTT):** A method for computing gradients in RNNs by unrolling the network over time.
- **LSTM:** An RNN variant with gates to control the flow of information, designed to capture long-range dependencies.
- **GRU:** A simpler version of LSTM that uses fewer gates while still addressing the vanishing gradient problem.
- **Encoder-Decoder:** An architecture for mapping an input sequence to an output sequence.
- **Attention Mechanism:** Allows the decoder to selectively focus on parts of the input sequence during decoding.

---

## Lecture 8: Neural Networks and Backpropagation

**3Blue1Brown Video:**  
[Neural Networks – Backpropagation](https://www.youtube.com/watch?v=2vI82f-0qxI)

### Overview & Key Topics
- **Vanishing Gradient Problem:** Detailed explanation of why gradients diminish in deep networks, especially with saturating activations.
- **Dying ReLU and Leaky ReLU:** Explanation of how standard ReLU can lead to inactive neurons and how Leaky ReLU mitigates this issue.
- **Momentum:** How momentum accumulates gradient history to improve convergence and avoid local minima.
- **Dropout Regularisation:** Method to reduce overfitting by randomly deactivating neurons during training.
- **Early Stopping:** Monitoring validation performance to stop training before overfitting.
- **Weight Sharing:** How CNNs and RNNs use weight sharing to reduce parameters and promote invariant feature learning.
- **L1 vs L2 Regularisation:** How these techniques affect model sparsity and weight distribution.
- **Data Augmentation & Noise Injection:** Techniques to improve generalisation.
- **Transfer Learning:** How pre-trained models (e.g., on ImageNet or large corpora) can be fine-tuned for new tasks.

### Glossary
- **Backpropagation:** The algorithm to compute gradients and update weights in a neural network.
- **Momentum:** A term that accumulates past gradients to stabilize updates.
- **Dropout:** A technique that randomly turns off neurons during training.
- **Weight Sharing:** Reusing the same weights across different parts of the network to reduce parameters.
- **L1 Regularisation:** Penalizes the sum of absolute weights, promoting sparsity.
- **L2 Regularisation:** Penalizes the sum of squared weights, preventing large weight values.
- **Data Augmentation:** Methods to expand training data by applying transformations.
- **Transfer Learning:** Reusing a pre-trained model on a new task.

---

## Lecture 9: Attention Mechanisms, Transformers, and Advanced Topics

**3Blue1Brown Video:**  
[Transformers and Attention](https://www.youtube.com/watch?v=7VeUPuFGJHk)

### Overview & Key Topics
- **Attention Mechanisms:** Concept and benefits – allowing models to focus on relevant parts of the input.  
- **Query-Key-Value Design:** How attention computes compatibility scores to weight different parts of the input.
- **Self-Attention:** Allowing elements of a sequence to interact directly with each other, enabling modeling of long-range dependencies without recurrence.
- **Transformer Architecture:** A deep learning model based entirely on attention mechanisms (multi-head attention, positional encoding, feed-forward networks, residual connections, and layer normalization) without recurrent or convolutional components.
- **Positional Encoding:** Why and how positional information is injected into Transformer inputs.
- **Beam Search:** The search algorithm used to generate sequences in models like Transformers.
- **BERT and XLNet:** Examples of large pre-trained Transformer models and how they achieve state-of-the-art performance in NLP.
- **Applications and Implications:** How attention and Transformers have transformed fields such as machine translation, summarization, and language modeling.

### Glossary
- **Attention Mechanism:** A method that computes a weighted sum of input features where weights reflect the importance of each feature for the current output.
- **Self-Attention:** Attention applied within a single sequence to capture intra-sequence relationships.
- **Transformer:** A neural network architecture based entirely on attention, with no recurrent or convolutional components.  
- **Query, Key, Value:** In attention, the query is what you’re comparing, the keys represent the elements to compare against, and the values are the information that is combined according to the attention weights.
- **Multi-Head Attention:** Simultaneously computing multiple attention outputs from different learned projections to capture different aspects of relationships.
- **Positional Encoding:** A method to incorporate sequence order into Transformer inputs.
- **Beam Search:** A heuristic search strategy that explores multiple candidate sequences to generate the most likely output.
- **BERT:** A pre-trained Transformer model that provides deep bidirectional representations, widely used for NLP tasks.

---

## Additional Resources & Final Thoughts

### Additional Resources
- **Textbooks & Papers:**
  - [Chris Olah's Blog](http://colah.github.io/)
  - [book by Michael Nielsen](http://neuralnetworksanddeeplearning)
  - *Deep Learning* by Goodfellow, Bengio, and Courville.
  - *Attention Is All You Need* (Vaswani et al., 2017) for Transformers.
  - ResNet, GAN, and GCN foundational papers.
- **Online Courses:**
  - Stanford CS231n (Convolutional Neural Networks)
  - Stanford CS224n (NLP with Deep Learning)
  - fast.ai Deep Learning courses
- **Code Repositories:**
  - [PyTorch Examples](https://github.com/pytorch/examples)
  - [TensorFlow Models](https://github.com/tensorflow/models)
  - [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **Deep Learning Frameworks:**
  - PyTorch, TensorFlow, JAX
  - Hugging Face Hub for pre-trained models

### Final Thoughts
Deep learning has transformed how we approach complex problems in computer vision, natural language processing, and beyond. This course has equipped you with advanced tools and theoretical insights—from optimization and regularization strategies to innovative architectures like Transformers and GNNs. As you continue to explore this field, remember that continuous experimentation and critical evaluation of models are key. Stay curious, engage with the latest research, and always consider the ethical implications of deploying these powerful techniques in real-world applications.

Happy learning, and may your journey in deep learning lead to innovative breakthroughs!

---
## Glossary of Key Terms

Activation Function: Applies non-linearity (e.g., sigmoid outputs between 0 and 1).
Attention Mechanism: A method that computes a weighted sum of input features where weights reflect the importance of each feature for the current output.
Backpropagation: The algorithm to compute gradients and update weights in a neural network.
Backpropagation Through Time (BPTT): A method for computing gradients in RNNs by unrolling the network over time.
Beam Search: A heuristic search strategy that explores multiple candidate sequences to generate the most likely output.
BERT: A pre-trained Transformer model that provides deep bidirectional representations, widely used for NLP tasks.
Bias: A learnable parameter in neural networks that allows shifting the activation function.
Bidirectional RNN: Processes the input sequence in both forward and backward directions.
Binary Threshold Unit (BTU): The activation that outputs 1 if the weighted sum exceeds a threshold.
Bottleneck Layer: A layer in autoencoders that has fewer neurons than the previous layers, forcing a compressed representation.
Building Theorem: The concept that multilayer perceptrons can implement any Boolean function.
Chain Rule: A calculus rule that allows the derivative of a composite function to be computed as the product of derivatives.
Computational Graph: A graph representing the operations in a neural network, used for gradient computation.
Cone of Solutions: The set of weight vectors that yield correct classifications.
Convolution: A mathematical operation for feature extraction by sliding a filter over input data.
Cross-Entropy Loss: Measures the discrepancy between the predicted probability distribution and the true distribution.
Data Augmentation: Methods to expand training data by applying transformations.
Deep Networks: Networks with more than one hidden layer.
Delta (𝛿): The error signal for a neuron used in weight updates.
Dropout: A technique that randomly turns off neurons during training.
Epoch: One full pass through the training dataset.
Encoder-Decoder: An architecture for mapping an input sequence to an output sequence.
Filter/Kernel: A set of learnable weights in a CNN that extracts features from the input.
GAN (Generative Adversarial Network): Comprises a generator and discriminator competing adversarially.
Gradient Descent (GD): An optimization algorithm that updates parameters by moving in the direction of the negative gradient.
GRU: A simpler version of LSTM that uses fewer gates while still addressing the vanishing gradient problem.
Hebb's Rule: “Fire together, wire together” – strengthening of connections through simultaneous activation.
Hidden Layer: An intermediate layer in a multilayer perceptron that enables non-linear decision boundaries.
Hyperplane: A flat subspace of dimension n−1 in an 𝑛-dimensional space, used by perceptrons for linear separation.
ImageNet: A benchmark dataset with millions of labeled images.
Inception Module: A block that applies multiple convolutions of different sizes in parallel.
Information Capacity: The ratio of patterns learned to inputs available; beyond a threshold, learning fails.
L1 Regularisation: Penalizes the sum of absolute weights, promoting sparsity.
L2 Regularisation: Penalizes the sum of squared weights, preventing large weight values.
Learning Rate (λ): A hyperparameter that dictates the size of weight updates.
Linear Separability: The ability to divide classes using a straight line or hyperplane.
LSTM: An RNN variant with gates to control the flow of information, designed to capture long-range dependencies.
Momentum: A term that accumulates past gradients to stabilize updates.
Multi-Head Attention: Simultaneously computing multiple attention outputs from different learned projections to capture different aspects of relationships.
One-Hot Encoding: Represents categorical variables as binary vectors.
Perceptron: A basic neural network unit for binary classification.
Pooling: A downsampling operation that reduces spatial dimensions.
Positional Encoding: A method to incorporate sequence order into Transformer inputs.
PyTorch: A deep learning framework for building neural networks.
Query, Key, Value: In attention, the query is what you’re comparing, the keys represent the elements to compare against, and the values are the information that is combined according to the attention weights.
Receptive Field: The region in the input space that affects a particular neuron’s output.
ReLU: Outputs the input if positive, otherwise zero.
Residual Network (ResNet): A CNN with skip connections that help with training very deep networks.
RNN: A neural network that processes sequential data by maintaining and updating a hidden state.
Self-Attention: Attention applied within a single sequence to capture intra-sequence relationships.
Shallow Networks: Networks without hidden layers.
SoftMax Function: Converts a vector of logits into a probability distribution over classes.
Stochastic Gradient Descent (SGD): GD performed on a single example (or small batch), adding noise but speeding up computation.
Tanh: Outputs values between -1 and 1.
Transformer: A neural network architecture based entirely on attention, with no recurrent or convolutional components.
Transfer Learning: Reusing a pre-trained model on a new task.
Vanishing Gradient: A phenomenon where gradients diminish as they are propagated through many layers, slowing learning in early layers.
Visible Units: Neurons in the input and output layers.
Weight Sharing: Reusing the same weights across different parts of the network to reduce parameters.
Weight Initialization: Techniques (like Xavier or He initialization) used to set initial weights for stable training.
Weights: In artificial neural networks, weights are used to indicate the strength of a synapse. Synaptic weights are adjusted during learning, allowing the network to perform useful computations.
