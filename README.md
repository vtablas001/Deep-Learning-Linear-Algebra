# Deep Learning Linear Algebra

As I study the basics of Deep Learning from scratch, I wanted to create this repo to annotate in a well-structured fashion the foundational math and keep coming back to refresh this repo as I progress both in content and uderstanding of the topic. Getting the matrix calculus right on paper is extremely useful when coding the math later in from-scratch implementations, especially when building out applied computer vision systems tools.

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

A two-layer network (often called a Multi-Layer Perceptron with one hidden layer) introduces non-linearity and a second transformation. This allows the model to learn complex, non-linear patterns in data, such as distinguishing anomalies in medical imaging or tracking metrics.

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

## 7. The Backward Pass (Matrix Calculus)
To train the network, we must compute the gradients of the loss ($L$) with respect to our learnable parameters ($W$ and $b$). Assuming we receive an upstream gradient $dY = \frac{\partial L}{\partial Y}$ of shape $(N \times M)$ from the loss function, we apply the chain rule using matrix multiplication.

### Weight Gradient ($dW$)
The gradient of the weights is the dot product of the transposed input and the upstream gradient. This maps the error back to the dimensions of the weights:
$$dW = X^T \cdot dY \in \mathbb{R}^{D \times M}$$

### Bias Gradient ($db$)
The gradient of the bias is the sum of the upstream gradients across the batch dimension ($N$). Mathematically, this is the dot product of the transposed ones-vector and $dY$:
$$db = \mathbf{1}_N^T \cdot dY = \sum_{i=1}^N dY_i \in \mathbb{R}^{1 \times M}$$

### Input Gradient ($dX$)
To continue backpropagation to earlier layers, we calculate the gradient with respect to the input by taking the dot product of the upstream gradient and the transposed weight matrix:
$$dX = dY \cdot W^T \in \mathbb{R}^{N \times D}$$

---

## 8. Optimization Dynamics
Once the gradients are computed, we use them to update the weights. As an economist, I view these optimizers as systems balancing "historical trends" (momentum) with "market volatility" (adaptive learning rates). Let $\alpha$ be the learning rate.

### 8.1 Stochastic Gradient Descent (SGD) + Momentum
Momentum accumulates a velocity vector ($v$) to smooth out noisy gradients and accelerate through flat regions of the loss landscape (where $\rho$ is the friction/decay rate, usually $0.9$):
$$v_{t+1} = \rho v_t - \alpha dW$$
$$W_{t+1} = W_t + v_{t+1}$$

### 8.2 RMSProp
RMSProp uses an exponentially decaying average of squared gradients ($cache$) to adaptively scale the learning rate per parameter. It punishes highly volatile weights and boosts stable ones:
$$cache_{t+1} = \rho \cdot cache_t + (1 - \rho) \cdot dW^2$$
$$W_{t+1} = W_t - \frac{\alpha \cdot dW}{\sqrt{cache_{t+1}} + \epsilon}$$

### 8.3 Adam (Adaptive Moment Estimation)

Adam is the hybrid workhorse for deep networks, combining the directional velocity of Momentum with the adaptive scaling of RMSProp. It is highly robust and often the default choice for training complex architectures because it handles sparse gradients and noisy data exceptionally well.

To understand Adam, we track two distinct "moments" of the gradient over time step $t$. Let $\alpha$ be our learning rate.

#### Step 1: The Exponential Moving Averages (The Moments)
First, we calculate the moving averages of the gradient and its square. 
* **The First Moment ($m_t$):** This acts like Momentum. It estimates the mean (first moment) of the gradients, tracking the general "trend" or direction.
* **The Second Moment ($v_t$):** This acts like RMSProp. It estimates the uncentered variance (second moment) of the gradients, tracking the "volatility" or scale of the updates.

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) dW$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (dW^2)$$

#### Step 2: Bias Correction
Because the moving averages $m$ and $v$ are initialized as vectors of zeros, they are heavily biased toward zero during the initial time steps. If we updated our weights using the raw $m_t$ and $v_t$, the network would take artificially small steps at the very beginning of training.

To fix this, Adam applies a **Bias Correction** based on the current iteration/time step $t$:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

*Mathematical Intuition:* Notice that in the first few iterations, $\beta^t$ is close to $1$, so the denominator $(1 - \beta^t)$ is very small. Dividing by a tiny fraction scales the $m$ and $v$ vectors up, counteracting the zero-initialization. As $t$ grows large (later in training), $\beta^t$ approaches $0$, the denominator approaches $1$, and the bias correction gracefully turns itself off.

#### Step 3: The Weight Update
Finally, we use the bias-corrected moments to update the parameters. We step in the direction of the trend ($\hat{m}_t$), but we scale the step size down for highly volatile weights by dividing by the square root of $\hat{v}_t$. The $\epsilon$ term is a tiny constant added to the denominator strictly to prevent a division by zero error.

$$W_t = W_{t-1} - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

#### Standard Hyperparameters
When implementing Adam from scratch, the industry-standard default values established in the original Kingma & Ba paper are:
* $\alpha = 0.001$ (Learning rate)
* $\beta_1 = 0.9$ (Decay rate for the first moment)
* $\beta_2 = 0.999$ (Decay rate for the second moment)
* $\epsilon = 10^{-8}$ (Numerical stability constant)

## 9. The Full Lifecycle: Forward Pass to Weight Update

When building out custom architectures for my applied data science and computer vision projects, understanding the entire lifecycle of a neural network is non-negotiable. The process is a continuous loop of three phases: the **Forward Pass**, **Backpropagation**, and the **Parameter Update**.

Below is the complete algebraic breakdown for a standard Two-Layer Neural Network.

### 9.1 The Forward Pass
The forward pass is where the model makes its prediction. We push the input matrix $X$ through our learnable weights and non-linearities to get the final raw scores (logits).

1. **First Linear Transformation:** $$Z_1 = X W_1 + \mathbf{1}_N b_1$$
2. **Non-linear Activation (ReLU):** $$A_1 = \max(0, Z_1)$$
3. **Second Linear Transformation (Scores):** $$Z_2 = A_1 W_2 + \mathbf{1}_N b_2$$
4. **The Loss Function ($L$):** We evaluate how far off our scores $Z_2$ are from the ground truth $Y_{true}$. The loss collapses our high-dimensional predictions into a single, measurable scalar value:
   $$L = \text{Loss}(Z_2, Y_{true})$$

---

## 10. Backpropagation: Gradients and The Jacobian

To improve the model, I need to know how every single weight in $W_1$ and $W_2$ contributed to the final scalar error $L$. This is where the **Gradient** and the **Jacobian Matrix** come into play.

### The Jacobian Matrix
In vector calculus, if we have a function that maps an $n$-dimensional vector to an $m$-dimensional vector, the derivative of that function is the **Jacobian Matrix** (an $m \times n$ matrix of all partial derivatives). 

For example, the local derivative of our hidden layer $Z_1$ with respect to the input $X$ is technically a massive Jacobian matrix. However, because our final Loss $L$ is a scalar ($m=1$), the final chain-rule derivative we care about is the **Gradient**—a vector (or matrix) pointing in the direction of the steepest ascent of the loss. 

### The Vector-Jacobian Product (VJP)
When writing this in code, constructing a full Jacobian matrix for millions of parameters would crash the system's memory. Instead, backpropagation relies on the **Vector-Jacobian Product**. We take the "upstream gradient" (how much the loss cares about the output of a layer) and multiply it by the local gradient (the Jacobian of that specific layer) to get the "downstream gradient" to pass backward.

### 10.1 The Algebra of Backpropagation (Two-Layer Net)
Using the chain rule, we compute the gradients starting from the end of the network and moving backward.

**1. Gradient at the Output:**
We start with the derivative of the loss with respect to our raw scores.
$$dZ_2 = \frac{\partial L}{\partial Z_2}$$

**2. Gradients of Layer 2 Parameters:**
We use $dZ_2$ to find how to change $W_2$ and $b_2$. The gradient of the weights is the dot product of the transposed incoming activation and the upstream gradient.
$$dW_2 = A_1^T \cdot dZ_2$$
$$db_2 = \sum dZ_2 \quad \text{(summed across the batch dimension)}$$

**3. Passing the Gradient Backward (Input to Layer 2):**
To keep moving backward, we need the gradient with respect to the activation $A_1$.
$$dA_1 = dZ_2 \cdot W_2^T$$

**4. The ReLU Local Jacobian:**
The derivative of the ReLU function $A_1 = \max(0, Z_1)$ is $1$ if $Z_1 > 0$, and $0$ otherwise. Instead of a full Jacobian, we apply this as an element-wise multiplication (Hadamard product, denoted by $\odot$) with a binary mask.
$$dZ_1 = dA_1 \odot \mathbb{1}(Z_1 > 0)$$

**5. Gradients of Layer 1 Parameters:**
Finally, we compute the gradients for our first layer's weights and biases.
$$dW_1 = X^T \cdot dZ_1$$
$$db_1 = \sum dZ_1$$



