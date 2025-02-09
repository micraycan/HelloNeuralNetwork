using System;
using Unity.Mathematics;
using Random = System.Random;

namespace SVGL
{
    /// <summary>
    /// Represents a single layer in a neural network.
    /// Each layer consists of a number of neurons that have associated weights, biases, and outputs.
    /// The layer uses an activation function to produce its output values.
    /// </summary>
    [Serializable]
    public class Layer
    {
        /// <summary>
        /// Gets or sets the weight matrix of the layer.
        /// Each row corresponds to a neuron, and each column corresponds to an input connection.
        /// </summary>
        public float[,] Weights { get; set; }

        /// <summary>
        /// Gets or sets the biases for each neuron in the layer.
        /// </summary>
        public float[] Biases { get; set; }

        /// <summary>
        /// Gets or sets the output values for each neuron after applying the activation function.
        /// </summary>
        public float[] Outputs { get; set; }

        private int _inputSize;
        private int _numNeurons;
        private IActivationFunction _activation;
        private Random _random = new Random();

        /// <summary>
        /// Initializes a new instance of the <see cref="Layer"/> class with the specified parameters.
        /// </summary>
        /// <param name="inputSize">The number of inputs each neuron receives.</param>
        /// <param name="numNeurons">The number of neurons in this layer.</param>
        /// <param name="activation">The activation function to apply to the neuron outputs.</param>
        public Layer(int inputSize, int numNeurons, IActivationFunction activation)
        {
            _inputSize = inputSize;
            _numNeurons = numNeurons;
            _activation = activation;

            InitializeWeights();
        }

        /// <summary>
        /// Initializes the weights and biases for the layer.
        /// Weights are set to small random values, and biases are initialized to zero.
        /// </summary>
        private void InitializeWeights()
        {
            Weights = new float[_numNeurons, _inputSize];
            Biases = new float[_numNeurons];
            Outputs = new float[_numNeurons];

            for (int i = 0; i < _numNeurons; i++)
            {
                for (int j = 0; j < _inputSize; j++)
                {
                    Weights[i, j] = (float)(_random.NextDouble() * 0.1 - 0.05);
                }

                Biases[i] = 0;
            }
        }

        /// <summary>
        /// Performs a forward pass through the layer.
        /// For each neuron, computes the weighted sum of inputs plus bias, applies the activation function, and stores the output.
        /// </summary>
        /// <param name="inputs">An array of input values to the layer.</param>
        /// <returns>An array of output values after activation.</returns>
        public float[] Forward(float[] inputs)
        {
            for (int i = 0; i < _numNeurons; i++)
            {
                float sum = Biases[i];

                // compute the weighted sum of inputs
                for (int j = 0; j < _inputSize; j++)
                {
                    sum += Weights[i, j] * inputs[j];
                }

                // apply activation function
                Outputs[i] = _activation.Activate(sum);
            }

            return Outputs;
        }

        /// <summary>
        /// Performs a backward pass through the layer, updating weights and biases based on the error.
        /// Returns the error to propagate to the previous layer.
        /// </summary>
        /// <param name="inputs">The original input values that were fed into this layer.</param>
        /// <param name="error">The error gradient received from the next layer.</param>
        /// <param name="learningRate">The learning rate used to scale the weight and bias updates.</param>
        /// <returns>An array of error gradients to be propagated to the previous layer.</returns>
        public float[] Backward(float[] inputs, float[] error, float learningRate)
        {
            float[] newError = new float[_inputSize];

            for (int i = 0; i < _numNeurons; i++)
            {
                // product of the error and derivative of the activation function evaluated at output
                float delta = error[i] * _activation.Derivative(Outputs[i]);
                Biases[i] -= learningRate * delta;

                // update weights and accumulate error for previous layer
                for (int j = 0; j < _inputSize; j++)
                {
                    newError[j] += Weights[i, j] * delta;
                    Weights[i, j] -= learningRate * delta * inputs[j];
                }
            }

            return newError;
        }
    }
}
