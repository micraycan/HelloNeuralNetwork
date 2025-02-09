using System;
using Unity.Mathematics;
using Random = System.Random;

namespace SVGL
{
    [Serializable]
    public class Layer
    {
        public float[,] Weights { get; set; }
        public float[] Biases { get; set; }
        public float[] Outputs { get; set; }

        private int _inputSize;
        private int _numNeurons;
        private IActivationFunction _activation;
        private Random _random = new Random();

        public Layer(int inputSize, int numNeurons, IActivationFunction activation)
        {
            _inputSize = inputSize;
            _numNeurons = numNeurons;
            _activation = activation;

            InitializeWeights();
        }

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

        public float[] Forward(float[] inputs)
        {
            for (int i = 0; i < _numNeurons; i++)
            {
                float sum = Biases[i];

                for (int j = 0; j < _inputSize; j++)
                {
                    sum += Weights[i, j] * inputs[j];
                }

                Outputs[i] = _activation.Activate(sum);
            }

            return Outputs;
        }

        public float[] Backward(float[] inputs, float[] error, float learningRate)
        {
            float[] newError = new float[_inputSize];

            for (int i = 0; i < _numNeurons; i++)
            {
                float delta = error[i] * _activation.Derivative(Outputs[i]);
                Biases[i] -= learningRate * delta;

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
