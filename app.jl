using Zygote
using LinearAlgebra

# Linear transformation
linear(W, b, x) = W * x + b

# Relu activation
relu(x) = max(0, x)

# Layer with activation
layer(W, b, activation, x) = activation.(linear(W, b, x))

# Dimensionalities of the layers
L = [3, 5, 2]
weights_0 = []

# Initialize the weights
for (D_i, D_i1) in zip(L[1:length(L)-1], L[2:length(L)])
	W_0 = rand(D_i1,D_i) - ones(D_i1,D_i) * 0.5
	b_0 = rand(D_i1) - ones(D_i1) * 0.5
	push!(weights_0,(W_0, b_0))
end

# Define neural network
function y_hat(x, weights)
	for (W, b) in weights
		x = layer(W, b, relu, x)
	end
	x
end

# Test drive
x_0 = rand(L[1])
println(y_hat(x_0, weights_0))

# Define target data point and loss
y = rand(L[length(L)])
loss(x, weights) = LinearAlgebra.norm(y_hat(x, weights) - y)

# Calculate and print gradient
g = gradient(loss, x_0, weights_0)
println("Gradient: ", g)