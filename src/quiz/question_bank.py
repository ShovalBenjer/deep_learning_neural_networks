from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random


class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    CODE_COMPLETION = "code_completion"
    CONCEPT_EXPLANATION = "concept_explanation"


@dataclass
class Question:
    id: str
    type: QuestionType
    section: str
    topic: str
    difficulty: str
    question_text: str
    options: Optional[Dict[str, str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    code_template: Optional[str] = None
    blank_description: Optional[str] = None
    key_concepts: Optional[List[str]] = None
    grading_rubric: Optional[List[str]] = None
    sample_answer: Optional[str] = None


SECTIONS = {
    "foundations": {
        "title": "I. Foundations of Neural Networks",
        "topics": [
            "What are Neural Networks",
            "Biological Inspiration: The Neuron",
            "Artificial Neurons: Building Blocks of ANNs",
            "Network Architecture: Visible, Hidden, and Depth",
            "AI Spectrum: Weak, General, and Strong AI",
            "Perceptron Model",
            "Binary Threshold Unit (BTU) Activation",
            "Mathematical Representation of BTU",
            "Hebb's Rule and Perceptron Learning",
            "Implementing Logic Gates with Perceptrons",
            "Linear Separability and Decision Boundaries",
            "Limitations of Single-Layer Perceptrons",
            "Types of Activation Functions",
            "Logistic Sigmoid Neuron",
            "ReLU and Leaky ReLU",
            "Hyperbolic Tangent (Tanh)",
            "Stochastic Binary Neuron",
            "Choosing the Right Activation Function",
            "Supervised vs Unsupervised Learning",
            "Perceptron Learning Algorithm",
            "Information Capacity and Learning Limits",
        ],
    },
    "training": {
        "title": "II. Training Deep Neural Networks",
        "topics": [
            "Gradient Descent Concept",
            "Batch Gradient Descent",
            "Stochastic Gradient Descent (SGD)",
            "Mini-Batch Gradient Descent",
            "Learning Rate: Importance and Adaptation",
            "Momentum in Gradient Descent",
            "Forward Pass: Computing Network Output",
            "Backward Pass: Propagating Error Gradients",
            "Delta Values: Output and Hidden Layers",
            "Computational Graphs for Backpropagation",
            "Mean Squared Error (MSE)",
            "Cross-Entropy Loss",
            "Choosing the Right Cost Function",
        ],
    },
    "challenges": {
        "title": "III. Challenges in Deep Learning",
        "topics": [
            "Understanding Overfitting",
            "L1 and L2 Regularization",
            "Dropout Regularization",
            "Early Stopping",
            "Data Augmentation and Noise Injection",
            "Batch Normalization",
            "Vanishing Gradient Problem",
            "Exploding Gradient Problem",
            "Weight Initialization Strategies",
            "Gradient Clipping",
            "Local Minima Problem",
            "Adaptive Learning Rates (Adam, AdaGrad)",
            "Ensemble Methods",
        ],
    },
    "architectures": {
        "title": "IV. Deep Learning Architectures",
        "topics": [
            "Hierarchical Feature Learning",
            "Autoencoders: Unsupervised Feature Learning",
            "Convolutional Layers: Feature Extraction",
            "Parameter Sharing and Local Connectivity",
            "Pooling Layers: Downsampling and Invariance",
            "CNN Architectures: ResNet and Inception",
            "Recurrent Neural Networks (RNNs)",
            "Hidden State and Temporal Context",
            "Backpropagation Through Time (BPTT)",
            "Bidirectional RNNs",
            "LSTM Architecture",
            "GRU Architecture",
            "LSTM vs GRU Comparison",
            "Attention Mechanism Principle",
            "Query, Key, Value Attention Design",
            "Self-Attention",
            "Multi-Head Attention",
            "Positional Encoding",
            "Transformer Architecture",
            "BERT and XLNet",
        ],
    },
    "evaluation": {
        "title": "V. Model Evaluation",
        "topics": [
            "Accuracy, Error Rate, and Loss",
            "Precision, Recall, and F1 Score",
            "Specificity and Sensitivity",
            "Confusion Matrix",
            "Choosing Appropriate Metrics",
        ],
    },
}


STATIC_QUESTIONS: List[Question] = [
    Question(
        id="foundations-mc-001",
        type=QuestionType.MULTIPLE_CHOICE,
        section="foundations",
        topic="What are Neural Networks",
        difficulty="easy",
        question_text="What is the primary inspiration behind artificial neural networks?",
        options={
            "A": "Quantum mechanics",
            "B": "Biological neural networks in the brain",
            "C": "Classical logic gates",
            "D": "Statistical regression models",
        },
        correct_answer="B",
        explanation="Artificial neural networks are computational models inspired by the structure and function of biological neural networks in the brain.",
    ),
    Question(
        id="foundations-mc-002",
        type=QuestionType.MULTIPLE_CHOICE,
        section="foundations",
        topic="Perceptron Model",
        difficulty="easy",
        question_text="Who introduced the Perceptron as one of the earliest neural network models?",
        options={
            "A": "Geoffrey Hinton",
            "B": "Frank Rosenblatt",
            "C": "Yann LeCun",
            "D": "Andrew Ng",
        },
        correct_answer="B",
        explanation="Frank Rosenblatt introduced the Perceptron in the late 1950s as one of the earliest and simplest types of artificial neural networks.",
    ),
    Question(
        id="foundations-mc-003",
        type=QuestionType.MULTIPLE_CHOICE,
        section="foundations",
        topic="Binary Threshold Unit (BTU) Activation",
        difficulty="medium",
        question_text="What is the output of a Binary Threshold Unit (BTU) when the weighted sum z_i = 0.5 and the threshold is 0?",
        options={
            "A": "0.5",
            "B": "0",
            "C": "1",
            "D": "Undefined",
        },
        correct_answer="C",
        explanation="BTU outputs 1 when z_i > 0 (threshold = 0). Since 0.5 > 0, the output is 1.",
    ),
    Question(
        id="foundations-mc-004",
        type=QuestionType.MULTIPLE_CHOICE,
        section="foundations",
        topic="Linear Separability and Decision Boundaries",
        difficulty="medium",
        question_text="Which logic gate CANNOT be implemented by a single-layer perceptron?",
        options={
            "A": "AND",
            "B": "OR",
            "C": "NOT",
            "D": "XOR",
        },
        correct_answer="D",
        explanation="XOR is not linearly separable, so a single-layer perceptron cannot implement it. This is a fundamental limitation of single-layer perceptrons.",
    ),
    Question(
        id="foundations-mc-005",
        type=QuestionType.MULTIPLE_CHOICE,
        section="foundations",
        topic="Types of Activation Functions",
        difficulty="medium",
        question_text="Which activation function helps mitigate the vanishing gradient problem for positive inputs?",
        options={
            "A": "Sigmoid",
            "B": "Tanh",
            "C": "ReLU",
            "D": "Binary Threshold Unit",
        },
        correct_answer="C",
        explanation="ReLU (Rectified Linear Unit) mitigates the vanishing gradient problem for positive inputs because its gradient is always 1 for positive values, unlike Sigmoid and Tanh which saturate.",
    ),
    Question(
        id="foundations-mc-006",
        type=QuestionType.MULTIPLE_CHOICE,
        section="foundations",
        topic="Types of Activation Functions",
        difficulty="hard",
        question_text="What is the primary disadvantage of the ReLU activation function?",
        options={
            "A": "It is computationally expensive",
            "B": "It suffers from the vanishing gradient problem for positive inputs",
            "C": "Neurons can 'die' when inputs are consistently negative",
            "D": "Its output is not zero-centered for positive inputs",
        },
        correct_answer="C",
        explanation="The 'dying ReLU' problem occurs when a neuron's weights are updated such that its input is always negative. The gradient for negative inputs is zero, so the neuron stops learning entirely.",
    ),
    Question(
        id="foundations-mc-007",
        type=QuestionType.MULTIPLE_CHOICE,
        section="foundations",
        topic="Supervised vs Unsupervised Learning",
        difficulty="easy",
        question_text="In which type of learning does the model learn from labeled data with input-output pairs?",
        options={
            "A": "Unsupervised Learning",
            "B": "Supervised Learning",
            "C": "Reinforcement Learning",
            "D": "Transfer Learning",
        },
        correct_answer="B",
        explanation="In supervised learning, the model learns from labeled data consisting of input-output pairs, where each input is associated with a correct output label.",
    ),
    Question(
        id="training-mc-001",
        type=QuestionType.MULTIPLE_CHOICE,
        section="training",
        topic="Gradient Descent Concept",
        difficulty="easy",
        question_text="What is the primary goal of gradient descent in training neural networks?",
        options={
            "A": "Maximize the cost function",
            "B": "Minimize the cost function",
            "C": "Maximize the learning rate",
            "D": "Increase the number of parameters",
        },
        correct_answer="B",
        explanation="Gradient descent minimizes the cost function (loss) by iteratively moving in the direction of the steepest decrease, which is the negative gradient direction.",
    ),
    Question(
        id="training-mc-002",
        type=QuestionType.MULTIPLE_CHOICE,
        section="training",
        topic="Stochastic Gradient Descent (SGD)",
        difficulty="medium",
        question_text="How does Stochastic Gradient Descent (SGD) differ from Batch Gradient Descent?",
        options={
            "A": "SGD uses the entire dataset for each update",
            "B": "SGD updates weights after each individual training example",
            "C": "SGD always converges faster than Batch GD",
            "D": "SGD never converges to the minimum",
        },
        correct_answer="B",
        explanation="SGD computes the gradient and updates weights for each training example individually, making it faster per iteration but more erratic in convergence compared to Batch GD.",
    ),
    Question(
        id="training-mc-003",
        type=QuestionType.MULTIPLE_CHOICE,
        section="training",
        topic="Learning Rate: Importance and Adaptation",
        difficulty="medium",
        question_text="What happens when the learning rate is set too large during gradient descent?",
        options={
            "A": "The model converges too quickly to the optimal solution",
            "B": "The algorithm may overshoot the minimum and diverge",
            "C": "The gradients become zero",
            "D": "The cost function decreases steadily",
        },
        correct_answer="B",
        explanation="A learning rate that is too large can cause the algorithm to overshoot the minimum, leading to oscillation or divergence, making the learning process unstable.",
    ),
    Question(
        id="training-mc-004",
        type=QuestionType.MULTIPLE_CHOICE,
        section="training",
        topic="Forward Pass: Computing Network Output",
        difficulty="medium",
        question_text="During the forward pass of a neural network, what is computed?",
        options={
            "A": "The gradient of the loss with respect to each weight",
            "B": "The output of the network given the current inputs and weights",
            "C": "The weight updates using backpropagation",
            "D": "The learning rate schedule",
        },
        correct_answer="B",
        explanation="The forward pass computes the network's output by propagating the input through each layer using the current weights and activation functions.",
    ),
    Question(
        id="training-mc-005",
        type=QuestionType.MULTIPLE_CHOICE,
        section="training",
        topic="Cross-Entropy Loss",
        difficulty="hard",
        question_text="Why is cross-entropy loss preferred over MSE for classification problems?",
        options={
            "A": "MSE cannot be computed for classification",
            "B": "Cross-entropy produces larger gradients when predictions are wrong, leading to faster learning",
            "C": "Cross-entropy always produces a value between 0 and 1",
            "D": "MSE is only for regression and cannot handle probabilities",
        },
        correct_answer="B",
        explanation="Cross-entropy loss produces larger gradients when predictions are very wrong compared to MSE, which can have very small gradients near saturation. This helps classification models learn faster from errors.",
    ),
    Question(
        id="challenges-mc-001",
        type=QuestionType.MULTIPLE_CHOICE,
        section="challenges",
        topic="Understanding Overfitting",
        difficulty="easy",
        question_text="What is overfitting in the context of neural networks?",
        options={
            "A": "When the model performs well on both training and test data",
            "B": "When the model performs well on training data but poorly on unseen test data",
            "C": "When the model is too simple to capture patterns in the data",
            "D": "When the training loss is zero",
        },
        correct_answer="B",
        explanation="Overfitting occurs when a model learns the training data too well, including noise, and fails to generalize to unseen data. It performs well on training data but poorly on test data.",
    ),
    Question(
        id="challenges-mc-002",
        type=QuestionType.MULTIPLE_CHOICE,
        section="challenges",
        topic="Dropout Regularization",
        difficulty="medium",
        question_text="How does dropout regularization work during training?",
        options={
            "A": "It adds noise to the input data",
            "B": "It randomly deactivates neurons during each training iteration",
            "C": "It increases the learning rate over time",
            "D": "It removes entire layers from the network",
        },
        correct_answer="B",
        explanation="Dropout randomly deactivates a fraction of neurons during each training iteration, forcing the network to learn redundant representations and preventing co-adaptation of features.",
    ),
    Question(
        id="challenges-mc-003",
        type=QuestionType.MULTIPLE_CHOICE,
        section="challenges",
        topic="Vanishing Gradient Problem",
        difficulty="medium",
        question_text="Which of the following activation functions is most susceptible to the vanishing gradient problem?",
        options={
            "A": "ReLU",
            "B": "Leaky ReLU",
            "C": "Sigmoid",
            "D": "Linear",
        },
        correct_answer="C",
        explanation="The Sigmoid function saturates for very large or very negative inputs, producing gradients close to zero. In deep networks, these small gradients compound through layers, causing the vanishing gradient problem.",
    ),
    Question(
        id="challenges-mc-004",
        type=QuestionType.MULTIPLE_CHOICE,
        section="challenges",
        topic="Weight Initialization Strategies",
        difficulty="hard",
        question_text="What is the purpose of Xavier/He initialization?",
        options={
            "A": "To make all weights equal at the start of training",
            "B": "To initialize weights so that the variance of activations is preserved across layers",
            "C": "To set all weights to zero for symmetry",
            "D": "To randomly set weights to very large values",
        },
        correct_answer="B",
        explanation="Xavier/He initialization sets the initial weights so that the variance of the activations remains approximately the same across layers, helping to prevent vanishing or exploding gradients at the start of training.",
    ),
    Question(
        id="challenges-mc-005",
        type=QuestionType.MULTIPLE_CHOICE,
        section="challenges",
        topic="Adaptive Learning Rates (Adam, AdaGrad)",
        difficulty="hard",
        question_text="What key improvement does the Adam optimizer introduce over standard SGD?",
        options={
            "A": "It uses only the first moment (mean) of gradients",
            "B": "It uses only the second moment (variance) of gradients",
            "C": "It combines momentum (first moment) with adaptive learning rates (second moment)",
            "D": "It eliminates the need for a learning rate hyperparameter",
        },
        correct_answer="C",
        explanation="Adam (Adaptive Moment Estimation) combines momentum (first moment estimate) with adaptive learning rates based on the second moment estimate of the gradients, providing faster and more stable convergence.",
    ),
    Question(
        id="architectures-mc-001",
        type=QuestionType.MULTIPLE_CHOICE,
        section="architectures",
        topic="Convolutional Layers: Feature Extraction",
        difficulty="medium",
        question_text="What is the key advantage of parameter sharing in convolutional layers?",
        options={
            "A": "It increases the number of trainable parameters",
            "B": "It reduces the number of parameters and enables translation invariance",
            "C": "It makes training slower but more accurate",
            "D": "It eliminates the need for pooling layers",
        },
        correct_answer="B",
        explanation="Parameter sharing in CNNs means the same filter is applied across different spatial locations, significantly reducing parameters and enabling translation invariance — detecting features regardless of their position.",
    ),
    Question(
        id="architectures-mc-002",
        type=QuestionType.MULTIPLE_CHOICE,
        section="architectures",
        topic="LSTM Architecture",
        difficulty="hard",
        question_text="What is the role of the forget gate in an LSTM cell?",
        options={
            "A": "To decide what new information to store in the cell state",
            "B": "To decide what information to discard from the cell state",
            "C": "To decide what output to produce from the cell",
            "D": "To compute the input activation",
        },
        correct_answer="B",
        explanation="The forget gate in an LSTM decides what information to discard from the cell state. It outputs values between 0 (completely forget) and 1 (completely keep) for each dimension of the cell state.",
    ),
    Question(
        id="architectures-mc-003",
        type=QuestionType.MULTIPLE_CHOICE,
        section="architectures",
        topic="Attention Mechanism Principle",
        difficulty="medium",
        question_text="What is the core idea behind the attention mechanism in neural networks?",
        options={
            "A": "Processing all inputs with equal weight",
            "B": "Focusing on the most relevant parts of the input by assigning different importance weights",
            "C": "Reducing the dimensionality of the input",
            "D": "Adding more layers to the network",
        },
        correct_answer="B",
        explanation="The attention mechanism allows the model to focus on different parts of the input by assigning different importance weights, enabling it to selectively attend to the most relevant information.",
    ),
    Question(
        id="architectures-mc-004",
        type=QuestionType.MULTIPLE_CHOICE,
        section="architectures",
        topic="Transformer Architecture",
        difficulty="hard",
        question_text="Why is positional encoding necessary in the Transformer architecture?",
        options={
            "A": "To reduce the computational complexity of attention",
            "B": "Because the self-attention mechanism has no inherent notion of sequence order",
            "C": "To normalize the input embeddings",
            "D": "To increase the model's vocabulary size",
        },
        correct_answer="B",
        explanation="Self-attention is permutation-invariant — it computes relationships between all positions equally without considering order. Positional encoding injects sequence order information so the model can distinguish between different positions.",
    ),
    Question(
        id="architectures-mc-005",
        type=QuestionType.MULTIPLE_CHOICE,
        section="architectures",
        topic="Recurrent Neural Networks (RNNs)",
        difficulty="medium",
        question_text="What is the key characteristic that distinguishes RNNs from feedforward networks?",
        options={
            "A": "RNNs have more layers",
            "B": "RNNs have connections that form directed cycles, allowing information to persist",
            "C": "RNNs use different activation functions",
            "D": "RNNs can only process fixed-length sequences",
        },
        correct_answer="B",
        explanation="RNNs have recurrent connections that form cycles, allowing them to maintain a hidden state that carries information from previous time steps. This enables them to process sequential data of variable length.",
    ),
    Question(
        id="evaluation-mc-001",
        type=QuestionType.MULTIPLE_CHOICE,
        section="evaluation",
        topic="Precision, Recall, and F1 Score",
        difficulty="medium",
        question_text="If a model has high precision but low recall, what does this indicate?",
        options={
            "A": "The model correctly identifies most positives but also many false positives",
            "B": "The model is very accurate when it predicts positive, but misses many actual positives",
            "C": "The model has balanced performance",
            "D": "The model predicts everything as negative",
        },
        correct_answer="B",
        explanation="High precision means most predicted positives are correct, while low recall means many actual positives are missed. The model is conservative — it only predicts positive when very confident but misses many true cases.",
    ),
    Question(
        id="evaluation-mc-002",
        type=QuestionType.MULTIPLE_CHOICE,
        section="evaluation",
        topic="Confusion Matrix",
        difficulty="easy",
        question_text="In a confusion matrix, what does the True Positive (TP) cell represent?",
        options={
            "A": "Cases where the model predicted negative and the actual was negative",
            "B": "Cases where the model predicted positive and the actual was positive",
            "C": "Cases where the model predicted positive but the actual was negative",
            "D": "Cases where the model predicted negative but the actual was positive",
        },
        correct_answer="B",
        explanation="True Positives (TP) are cases where the model correctly predicted the positive class — both the prediction and the actual label are positive.",
    ),
    Question(
        id="foundations-cc-001",
        type=QuestionType.CODE_COMPLETION,
        section="foundations",
        topic="Perceptron Model",
        difficulty="medium",
        question_text="Complete the implementation of a perceptron's forward pass that computes the weighted sum and applies a BTU activation function.",
        code_template="import numpy as np\n\ndef perceptron_forward(x, w, bias):\n    # Compute weighted sum\n    z = _____\n    # Apply BTU activation\n    output = _____\n    return output",
        blank_description="First blank: weighted sum of inputs and weights plus bias. Second blank: BTU activation (1 if z > 0, else 0).",
        correct_answer="np.dot(x, w) + bias; 1 if z > 0 else 0",
        explanation="The weighted sum z = np.dot(x, w) + bias computes the linear combination. The BTU activation outputs 1 when z > 0 and 0 otherwise.",
    ),
    Question(
        id="foundations-cc-002",
        type=QuestionType.CODE_COMPLETION,
        section="foundations",
        topic="Types of Activation Functions",
        difficulty="medium",
        question_text="Complete the implementation of common activation functions.",
        code_template="import numpy as np\n\ndef sigmoid(z):\n    return _____\n\ndef relu(z):\n    return _____\n\ndef tanh_activation(z):\n    return _____",
        blank_description="Implement sigmoid, ReLU, and tanh activation functions.",
        correct_answer="1 / (1 + np.exp(-z)); np.maximum(0, z); np.tanh(z)",
        explanation="Sigmoid: 1/(1+e^-z), ReLU: max(0,z), Tanh: (e^z - e^-z)/(e^z + e^-z) — these are the standard implementations.",
    ),
    Question(
        id="training-cc-001",
        type=QuestionType.CODE_COMPLETION,
        section="training",
        topic="Gradient Descent Concept",
        difficulty="hard",
        question_text="Complete the gradient descent weight update rule for a simple loss function.",
        code_template="import numpy as np\n\ndef gradient_descent_update(weights, gradients, learning_rate):\n    \"\"\"Update weights using gradient descent.\"\"\"\n    new_weights = _____\n    return new_weights",
        blank_description="Implement the gradient descent weight update: w_new = w_old - lr * gradient.",
        correct_answer="weights - learning_rate * gradients",
        explanation="The gradient descent update rule moves weights in the opposite direction of the gradient by a step size proportional to the learning rate: w_new = w - lr * ∇J(w).",
    ),
    Question(
        id="training-cc-002",
        type=QuestionType.CODE_COMPLETION,
        section="training",
        topic="Mean Squared Error (MSE)",
        difficulty="medium",
        question_text="Complete the implementation of the Mean Squared Error loss function.",
        code_template="import numpy as np\n\ndef mse_loss(y_pred, y_true):\n    \"\"\"Compute Mean Squared Error.\"\"\"\n    n = len(y_true)\n    loss = _____\n    return loss",
        blank_description="Compute MSE: average of squared differences between predictions and true values.",
        correct_answer="np.sum((y_pred - y_true) ** 2) / n",
        explanation="MSE computes the average of the squared differences: (1/n) * Σ(y_pred - y_true)², which penalizes larger errors more heavily.",
    ),
    Question(
        id="challenges-cc-001",
        type=QuestionType.CODE_COMPLETION,
        section="challenges",
        topic="L1 and L2 Regularization",
        difficulty="hard",
        question_text="Complete the L2 regularization term added to the loss function.",
        code_template="import numpy as np\n\ndef l2_regularized_loss(loss, weights, lambda_reg):\n    \"\"\"Add L2 regularization to the loss.\"\"\"\n    l2_term = _____\n    total_loss = loss + l2_term\n    return total_loss",
        blank_description="Compute L2 regularization: lambda/2 * sum of squared weights.",
        correct_answer="(lambda_reg / 2) * np.sum(np.square(weights))",
        explanation="L2 regularization adds (λ/2) * Σw² to the loss, penalizing large weights and promoting simpler models that generalize better.",
    ),
    Question(
        id="architectures-cc-001",
        type=QuestionType.CODE_COMPLETION,
        section="architectures",
        topic="Convolutional Layers: Feature Extraction",
        difficulty="hard",
        question_text="Complete a simple 2D convolution operation (no padding, stride=1).",
        code_template="import numpy as np\n\ndef conv2d(input_matrix, kernel):\n    \"\"\"Simple 2D convolution without padding, stride=1.\"\"\"\n    h, w = input_matrix.shape\n    kh, kw = kernel.shape\n    output = np.zeros((h - kh + 1, w - kw + 1))\n    for i in range(output.shape[0]):\n        for j in range(output.shape[1]):\n            output[i, j] = _____\n    return output",
        blank_description="Compute element-wise multiply and sum between input patch and kernel.",
        correct_answer="np.sum(input_matrix[i:i+kh, j:j+kw] * kernel)",
        explanation="Convolution computes the element-wise product of the input patch and kernel, then sums all values to produce each output element.",
    ),
    Question(
        id="architectures-cc-002",
        type=QuestionType.CODE_COMPLETION,
        section="architectures",
        topic="LSTM Architecture",
        difficulty="hard",
        question_text="Complete the LSTM cell computation for the forget gate and cell state update.",
        code_template="import numpy as np\n\ndef lstm_forget_gate(h_prev, x_t, Wf, bf):\n    \"\"\"Compute LSTM forget gate.\"\"\"\n    f_t = _____\n    return f_t",
        blank_description="Compute forget gate: sigmoid(Wf @ [h_prev, x_t] + bf).",
        correct_answer="sigmoid(np.dot(Wf, np.concatenate([h_prev, x_t])) + bf)",
        explanation="The forget gate computes f_t = σ(W_f · [h_{t-1}, x_t] + b_f), outputting values between 0 (forget) and 1 (keep) for each dimension of the cell state.",
    ),
    Question(
        id="foundations-ce-001",
        type=QuestionType.CONCEPT_EXPLANATION,
        section="foundations",
        topic="Linear Separability and Decision Boundaries",
        difficulty="medium",
        question_text="Explain why a single-layer perceptron cannot solve the XOR problem, and describe how this limitation is overcome.",
        key_concepts=["Linear separability", "Decision boundary", "Multi-layer networks", "Non-linear classification"],
        grading_rubric=[
            "Correctly identifies that XOR is not linearly separable",
            "Explains that single perceptron creates a linear decision boundary",
            "Describes how hidden layers in MLPs introduce non-linearity",
            "Mentions that MLPs can solve XOR by combining multiple linear boundaries",
        ],
        sample_answer="A single-layer perceptron can only create linear decision boundaries (hyperplanes), meaning it can only classify data that is linearly separable. The XOR function is not linearly separable — no single straight line can separate the positive and negative classes in XOR truth table. To overcome this, multi-layer perceptrons (MLPs) use hidden layers with non-linear activation functions. The hidden layer allows the network to create multiple linear boundaries that, when combined, can classify non-linearly separable data like XOR. Essentially, the hidden layer transforms the input into a new representation where the classes become linearly separable.",
    ),
    Question(
        id="foundations-ce-002",
        type=QuestionType.CONCEPT_EXPLANATION,
        section="foundations",
        topic="Types of Activation Functions",
        difficulty="hard",
        question_text="Compare and contrast Sigmoid, ReLU, and Tanh activation functions. Discuss when you would choose each one and their respective limitations.",
        key_concepts=["Vanishing gradient", "Non-linearity", "Zero-centered output", "Dying ReLU"],
        grading_rubric=[
            "Correctly describes mathematical properties of each function",
            "Discusses vanishing gradient problem for Sigmoid and Tanh",
            "Explains the dying ReLU problem",
            "Provides appropriate use cases for each function",
        ],
        sample_answer="Sigmoid outputs values between 0 and 1, making it suitable for binary classification output layers. However, it suffers from the vanishing gradient problem for extreme inputs and is not zero-centered. Tanh outputs between -1 and 1, is zero-centered (beneficial for convergence), but also suffers from vanishing gradients. ReLU outputs max(0, z), is computationally efficient, and mitigates vanishing gradients for positive inputs, making it the default choice for hidden layers. However, ReLU suffers from the 'dying ReLU' problem where neurons can permanently output zero for negative inputs. Use Sigmoid for binary output layers, Tanh when zero-centered output is beneficial (e.g., some RNNs), and ReLU as the default for hidden layers.",
    ),
    Question(
        id="training-ce-001",
        type=QuestionType.CONCEPT_EXPLANATION,
        section="training",
        topic="Backward Pass: Propagating Error Gradients",
        difficulty="hard",
        question_text="Explain the backpropagation algorithm, including how gradients are computed for hidden layers versus output layers, and why this enables training of deep networks.",
        key_concepts=["Chain rule", "Delta values", "Output layer gradient", "Hidden layer gradient", "Weight updates"],
        grading_rubric=[
            "Explains that backpropagation computes gradients using the chain rule",
            "Correctly describes delta computation for output vs hidden layers",
            "Explains how error signals propagate backward through the network",
            "Connects gradient computation to weight updates via gradient descent",
        ],
        sample_answer="Backpropagation computes gradients of the loss function with respect to each weight using the chain rule. For the output layer, the delta (error term) is computed directly from the loss derivative and the activation derivative. For hidden layers, the delta is computed by propagating the deltas from the subsequent layer backward, weighted by the connections, and multiplied by the activation derivative. This chain rule application allows error signals to flow from the output all the way back to earlier layers. The computed gradients are then used by gradient descent to update weights. This enables deep networks to learn because each layer receives a gradient signal telling it how to adjust to reduce the overall error.",
    ),
    Question(
        id="challenges-ce-001",
        type=QuestionType.CONCEPT_EXPLANATION,
        section="challenges",
        topic="Vanishing Gradient Problem",
        difficulty="hard",
        question_text="Explain the vanishing gradient problem in deep neural networks. Describe at least three strategies to mitigate it and explain how each strategy addresses the problem.",
        key_concepts=["Gradient saturation", "Activation function choice", "Weight initialization", "Residual connections", "Normalization"],
        grading_rubric=[
            "Correctly explains that gradients diminish exponentially through layers",
            "Describes at least three mitigation strategies",
            "Explains the mechanism by which each strategy addresses vanishing gradients",
            "Connects the strategies to practical implementation choices",
        ],
        sample_answer="The vanishing gradient problem occurs when gradients become extremely small as they are propagated backward through many layers, making it nearly impossible for earlier layers to learn. This happens because activation functions like Sigmoid and Tanh have derivatives less than 1, and multiplying many small values together causes exponential decay. Three mitigation strategies: (1) Using ReLU family activations — since ReLU's gradient is exactly 1 for positive inputs, gradients don't vanish through it; (2) Proper weight initialization (Xavier/He) — initializing weights so that variance is preserved across layers prevents gradients from shrinking; (3) Residual connections (ResNet) — skip connections provide gradient highways that bypass intermediate layers, ensuring gradient flow reaches earlier layers. Additional strategies include batch normalization and gradient clipping.",
    ),
    Question(
        id="architectures-ce-001",
        type=QuestionType.CONCEPT_EXPLANATION,
        section="architectures",
        topic="Transformer Architecture",
        difficulty="hard",
        question_text="Explain how the Transformer architecture processes input sequences without recurrence. Include the roles of self-attention, multi-head attention, and positional encoding.",
        key_concepts=["Self-attention mechanism", "Multi-head attention", "Positional encoding", "Parallel processing", "Query-Key-Value"],
        grading_rubric=[
            "Explains self-attention computation (Q, K, V mechanism)",
            "Describes multi-head attention and its purpose",
            "Explains positional encoding and why it's necessary",
            "Contrasts with RNN sequential processing, highlighting parallelism advantage",
        ],
        sample_answer="The Transformer processes sequences in parallel rather than sequentially like RNNs. Self-attention computes relationships between all positions simultaneously using Query, Key, and Value projections. For each position, attention scores are computed as softmax(QK^T/√d_k)V, allowing every position to attend to every other position. Multi-head attention runs multiple attention operations in parallel with different learned projections, allowing the model to capture different types of relationships (e.g., syntactic vs semantic). Positional encoding is necessary because self-attention is permutation-invariant — it has no notion of sequence order. Positional encodings (sinusoidal or learned) are added to input embeddings to inject position information. This parallelizable architecture enables much faster training than RNNs while capturing long-range dependencies effectively.",
    ),
]


class QuestionBank:
    def __init__(self):
        self.questions = {q.id: q for q in STATIC_QUESTIONS}
        self._topic_index: Dict[str, List[str]] = {}
        self._section_index: Dict[str, List[str]] = {}
        self._type_index: Dict[QuestionType, List[str]] = {}
        self._difficulty_index: Dict[str, List[str]] = {}
        self._build_indices()

    def _build_indices(self):
        for qid, q in self.questions.items():
            topic_key = f"{q.section}::{q.topic}"
            self._topic_index.setdefault(topic_key, []).append(qid)
            self._section_index.setdefault(q.section, []).append(qid)
            self._type_index.setdefault(q.type, []).append(qid)
            self._difficulty_index.setdefault(q.difficulty, []).append(qid)

    def get_question(self, question_id: str) -> Optional[Question]:
        return self.questions.get(question_id)

    def get_questions_by_section(self, section: str) -> List[Question]:
        qids = self._section_index.get(section, [])
        return [self.questions[qid] for qid in qids]

    def get_questions_by_topic(self, topic: str, section: str) -> List[Question]:
        topic_key = f"{section}::{topic}"
        qids = self._topic_index.get(topic_key, [])
        return [self.questions[qid] for qid in qids]

    def get_questions_by_type(self, qtype: QuestionType) -> List[Question]:
        qids = self._type_index.get(qtype, [])
        return [self.questions[qid] for qid in qids]

    def get_questions_by_difficulty(self, difficulty: str) -> List[Question]:
        qids = self._difficulty_index.get(difficulty, [])
        return [self.questions[qid] for qid in qids]

    def get_random_question(
        self,
        section: Optional[str] = None,
        topic: Optional[str] = None,
        qtype: Optional[QuestionType] = None,
        difficulty: Optional[str] = None,
    ) -> Optional[Question]:
        candidates = list(self.questions.values())
        if section:
            candidates = [q for q in candidates if q.section == section]
        if topic:
            candidates = [q for q in candidates if q.topic == topic]
        if qtype:
            candidates = [q for q in candidates if q.type == qtype]
        if difficulty:
            candidates = [q for q in candidates if q.difficulty == difficulty]
        if not candidates:
            return None
        return random.choice(candidates)

    def get_all_sections(self) -> Dict[str, Dict]:
        return SECTIONS

    def get_section_topics(self, section: str) -> List[str]:
        section_data = SECTIONS.get(section, {})
        return section_data.get("topics", [])

    def add_question(self, question: Question):
        self.questions[question.id] = question
        topic_key = f"{question.section}::{question.topic}"
        self._topic_index.setdefault(topic_key, []).append(question.id)
        self._section_index.setdefault(question.section, []).append(question.id)
        self._type_index.setdefault(question.type, []).append(question.id)
        self._difficulty_index.setdefault(question.difficulty, []).append(question.id)