# Linear transformation
linear(W, b, x) = W * x + b

# Relu activation
relu(x) = max(0, x)

# Layer with activation
layer(W, b, activation, x) = activation.(linear(W, b, x))

# Dimensionalities of the layers
L = [3, 5, 2]
weights = []

# Initialize the weights
for (D_i, D_i1) in zip(L[1:length(L)-1], L[2:length(L)])
	W_0 = rand(D_i1,D_i) - ones(D_i1,D_i) * 0.5
	b_0 = rand(D_i1) - ones(D_i1) * 0.5
	push!(weights,(W_0, b_0))
end

# Define neural network
function y_hat(x)
	for (W, b) in weights
		x = layer(W, b, relu, x)
	end
	x
end

# Test drive
x_0 = rand(L[1])
print(y_hat(x_0))