
# Linear transformation
linear(W, b, x) = W * x + b

# Relu activation
relu(x) = max(0, x)

# Layer with activation
layer(W, b, activation, x) = activation.(linear(W, b, x))


D = 4
L = 3
weights = []

for l in 0:L
	W_0 = rand(D,D) - ones(D,D) * 0.5
	b_0 = rand(D) - ones(D) * 0.5
	push!(weights,(W_0, b_0))
end

function y_hat(x)
	for (W, b) in weights
		x = layer(W, b, relu, x)
	end
	x
end

batch_size = 16
x_0 = rand(16, D)
print(y_hat.(x_0))
#y_hat(W1, W2, b1, b2, x)