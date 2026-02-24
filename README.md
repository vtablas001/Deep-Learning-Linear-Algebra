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

## 🛤️ Study Roadmap
- [x] Matrix Dimensions & Broadcasting
- [ ] Non-Linear Activation Functions (ReLU, Sigmoid)
- [ ] Chaining Layers (The 2-Layer Neural Network)
- [ ] Backpropagation Calculus
