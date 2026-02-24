# Deep Learning Linear Algebra

As I study Deep Learning from scratch, I wanted to create this repo to annotate in a well-structured fashion the basics, as it is super useful when coding the math later in assignment. This serves as a personal reference for bridging the gap between raw data tensors and formal algebraic operations.

## 1. The Image-to-Matrix Bridge
Since I am primarily working with computer vision, data in the form of images typically enters the system as a 4D tensor:

$$\mathcal{X} \in \mathbb{R}^{N \times H \times W \times C}$$

* **$N$**: Batch Size (number of images processed at once).
* **$H \times W$**: Spatial dimensions (Height and Width) of the image.
* **$C$**: Channels (e.g., 1 for grayscale, 3 for RGB).

### Flattening for Linear Layers
To use a standard fully connected layer, the spatial dimensions must be flattened into a single feature vector of length $D$:

$$D = H \times W \times C$$
$$X = \text{reshape}(\mathcal{X}, [N, D]) \in \mathbb{R}^{N \times D}$$

---

## 2. Learnable Parameters: Weights and Bias

### Weight Matrix ($W$)
The weight matrix maps the flattened input features to a specified number of output neurons ($M$):
$$W \in \mathbb{R}^{D \times M}$$

### Bias Vector ($b$)
The bias is a row vector providing a learnable offset for each output neuron:
$$b = \begin{bmatrix} b_1 & b_2 & \dots & b_M \end{bmatrix}_{1 \times M}$$

---

## 3. The Geometry of Broadcasting
In software, adding a $1 \times M$ vector to an $N \times M$ matrix happens via broadcasting. Mathematically, I represent this "replication" of the bias across the batch using a **column vector of ones** ($\mathbf{1}_N$).

### The Broadcasted Bias Vector to Matrix ($\mathbf{B}$)
To make the bias compatible for matrix addition with the weighted sum ($XW$), we define $\mathbf{B}$ as:

$$\mathbf{B} = \mathbf{1}_N b = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}_{N \times 1} \begin{bmatrix} b_1 & b_2 & \dots & b_M \end{bmatrix}_{1 \times M} = \begin{bmatrix} 
b_1 & b_2 & \dots & b_M \\ 
b_1 & b_2 & \dots & b_M \\ 
\vdots & \vdots & \ddots & \vdots \\ 
b_1 & b_2 & \dots & b_M 
\end{bmatrix}_{N \times M}$$

---

## 4. The Linear Transformation Equation
The complete algebraic representation of a single forward pass in a linear layer is:

$$Y = XW + \mathbf{1}_N b$$

### Dimension Consistency Check:
* **Weighted Sum ($XW$):** $(N \times D) \cdot (D \times M) = (N \times M)$
* **Broadcasted Bias ($\mathbf{1}_N b$):** $(N \times 1) \cdot (1 \times M) = (N \times M)$
* **Output ($Y$):** Resulting matrix is $(N \times M)$

---

## 5. The Two-Layer Neural Network

A two-layer network (often called a Multi-Layer Perceptron with one hidden layer) introduces non-linearity and a second transformation. This allows the model to learn complex, non-linear patterns in data, such as those found in medical imaging or tourist metrics.

### Mathematical Components
For this architecture, we define two sets of weights and biases:
* **Layer 1 (Hidden Layer):** $W_1 \in \mathbb{R}^{D \times H}$ and $b_1 \in \mathbb{R}^{1 \times H}$
* **Layer 2 (Output Layer):** $W_2 \in \mathbb{R}^{H \times M}$ and $b_2 \in \mathbb{R}^{1 \times M}$
* **Activation Function ($\sigma$):** Usually ReLU for the hidden layer.

### The Forward Pass Equations

1. **Hidden Layer Transformation ($Z_1$):**
   We calculate the weighted sum of the input and apply the broadcasted bias:
   $$Z_1 = XW_1 + \mathbf{1}_N b_1$$

2. **Activation ($A_1$):**
   We pass the hidden layer's output through a non-linear activation function (like ReLU):
   $$A_1 = \max(0, Z_1)$$

3. **Output Layer Transformation ($Y$):**
   The activated output $A_1$ now serves as the input for the final layer:
   $$Y = A_1 W_2 + \mathbf{1}_N b_2$$

### Full Chained Equation
$$Y = \sigma(XW_1 + \mathbf{1}_N b_1)W_2 + \mathbf{1}_N b_2$$

---

## 6. Dimension Tracking
Keeping track of dimensions is critical when coding these networks from scratch.

| Tensor | Description | Dimension |
| :--- | :--- | :--- |
| $X$ | Input Batch | $N \times D$ |
| $W_1$ | Hidden Weights | $D \times H$ |
| $b_1$ | Hidden Bias | $1 \times H$ |
| $A_1$ | Hidden Activation | $N \times H$ |
| $W_2$ | Output Weights | $H \times M$ |
| $b_2$ | Output Bias | $1 \times M$ |
| $Y$ | Final Output | $N \times M$ |

---
