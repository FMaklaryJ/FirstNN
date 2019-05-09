My first neural network.
Requires numpy and copy.

It tries to make an XOR gate with 2 neurons,
by using step functions as activation functions.
It is fully connected, so that each neuron is connected to every neuron in
every layer in front of it.

Enter the mutation rate (less than or equal to 1),
the amount of AI's per generation, and the number of generations,
and you're go.

If one wants to make it bigger, find the vector called NL at line 85.
The first element of NL is the number of neurons in the first layer,
the second element of NL is the number of neurons in the second layer,
the n'th element of NL is the number of neurons in the n'th layer.

The last element of NL is the output layer.