using System;
using System.IO;
using UnityEngine;

namespace SVGL
{
    [Serializable]
    public class NeuralNetwork
    {
        private int _inputSize;
        private int _hiddenSize1;
        private int _hiddenSize2;
        private int _outputSize;
        private float _learningRate;

        private float[,] _w1;
        private float[] _b1;
        private float[,] _w2;
        private float[] _b2;
        private float[,] _w3;
        private float[] _b3;

        System.Random rand = new System.Random();

        public NeuralNetwork(NetworkSettingsSO settings, bool randomValues = false)
        {
            _inputSize = settings.InputSize;
            _hiddenSize1 = settings.HiddenSize1;
            _hiddenSize2 = settings.HiddenSize2;
            _outputSize = settings.OutputSize;
            _learningRate = settings.LearningRate;

            _w1 = new float[_hiddenSize1, _inputSize];
            _b1 = new float[_hiddenSize1];
            _w2 = new float[_hiddenSize2, _hiddenSize1];
            _b2 = new float[_hiddenSize2];
            _w3 = new float[_outputSize, _hiddenSize2];
            _b3 = new float[_outputSize];

            if (randomValues)
            {
                for (int i = 0; i < _hiddenSize1; i++)
                {
                    for (int j = 0; j < _inputSize; j++)
                    {
                        _w1[i, j] = (float)(rand.NextDouble() * 0.1 - 0.05);
                    }

                    _b1[i] = 0;
                }

                for (int i = 0; i < _hiddenSize2; i++)
                {
                    for (int j = 0; j < _hiddenSize1; j++)
                    {
                        _w2[i, j] = (float)(rand.NextDouble() * 0.1 - 0.05);
                    }
                    _b2[i] = 0;
                }

                for (int i = 0; i < _outputSize; i++)
                {
                    for (int j = 0; j < _hiddenSize2; j++)
                    {
                        _w3[i, j] = (float)(rand.NextDouble() * 0.1 - 0.05);
                    }
                    _b3[i] = 0;
                }
            }
        }

        public float[] Forward(float[] input, out float[] h1, out float[] h2, out float[] logits)
        {
            h1 = new float[_hiddenSize1];
            for (int i = 0; i < _hiddenSize1; i++)
            {
                float sum = _b1[i];
                for (int j = 0; j < _inputSize; j++)
                {
                    sum += _w1[i, j] * input[j];
                }

                h1[i] = Utils.Sigmoid(sum);
            }

            h2 = new float[_hiddenSize2];
            for (int i = 0; i < _hiddenSize2; i++)
            {
                float sum = _b2[i];
                for (int j = 0; j < _hiddenSize1; j++)
                {
                    sum += _w2[i, j] * h1[j];
                }

                h2[i] = Utils.Sigmoid(sum);
            }

            logits = new float[_outputSize];
            for (int i = 0; i < _outputSize; i++)
            {
                float sum = _b3[i];

                for (int j = 0; j < _hiddenSize2; j++)
                {
                    sum += _w3[i, j] * h2[j];
                }

                logits[i] = sum;
            }

            float[] output = Utils.Softmax(logits);
            return output;
        }

        public void TrainOnSample(float[] input, int label)
        {
            float[] h1, h2, logits;
            float[] output = Forward(input, out h1, out h2, out logits);

            float[] target = new float[_outputSize];
            target[label] = 1;

            // compute error at output
            float[] errorOutput = new float[_outputSize];

            for (int i = 0; i < _outputSize; i++)
            {
                errorOutput[i] = output[i] - target[i];
            }

            float[,] gradientW3 = new float[_outputSize, _hiddenSize2];
            float[] gradientB3 = new float[_outputSize];

            for (int i = 0; i < _outputSize; i++)
            {
                gradientB3[i] = errorOutput[i];

                for (int j = 0; j < _hiddenSize2; j++)
                {
                    gradientW3[i, j] = errorOutput[i] * h2[j];
                }
            }

            float[] errorHidden2 = new float[_hiddenSize2];

            for (int j = 0; j < _hiddenSize2; j++)
            {
                float sum = 0;

                for (int i = 0; i < _outputSize; i++)
                {
                    sum += _w3[i, j] * errorOutput[i];
                }

                errorHidden2[j] = sum * Utils.SigmoidDerivative(h2[j]);
            }

            float[,] gradientW2 = new float[_hiddenSize2, _hiddenSize1];
            float[] gradientB2 = new float[_hiddenSize2];

            for (int i = 0; i < _hiddenSize2; i++)
            {
                gradientB2[i] = errorHidden2[i];
                for (int j = 0; j < _hiddenSize1; j++)
                {
                    gradientW2[i, j] = errorHidden2[i] * h1[j];
                }
            }

            float[] errorHidden1 = new float[_hiddenSize1];
            
            for (int j = 0; j < _hiddenSize1; j++)
            {
                float sum = 0;

                for (int i = 0; i < _hiddenSize2; i++)
                {
                    sum += _w2[i, j] * errorHidden2[i];
                }

                errorHidden1[j] = sum * Utils.SigmoidDerivative(h1[j]);
            }

            float[,] gradientW1 = new float[_hiddenSize1, _inputSize];
            float[] gradientB1 = new float[_hiddenSize1];

            for (int i = 0; i < _hiddenSize1; i++)
            {
                gradientB1[i] = errorHidden1[i];
                
                for (int j = 0; j < _inputSize; j++)
                {
                    gradientW1[i, j] = errorHidden1[i] * input[j];
                }
            }

            for (int i = 0; i < _hiddenSize1; i++)
            {
                for (int j = 0; j < _inputSize; j++)
                {
                    _w1[i, j] -= _learningRate * gradientW1[i, j];
                }

                _b1[i] -= _learningRate * gradientB1[i];
            }

            for (int i = 0; i < _hiddenSize2; i++)
            {
                for (int j = 0; j < _hiddenSize1; j++)
                {
                    _w2[i, j] -= _learningRate * gradientW2[i, j];
                }

                _b2[i] -= _learningRate * gradientB2[i];
            }

            for (int i = 0; i < _outputSize; i++)
            {
                for (int j = 0; j < _hiddenSize2; j++)
                {
                    _w3[i, j] -= _learningRate * gradientW3[i, j];
                }

                _b3[i] -= _learningRate * gradientB3[i];
            }
        }

        public void SaveWeights(string filePath)
        {
            NeuralNetworkWeightsData weightsData = new NeuralNetworkWeightsData();
            weightsData.InputSize = _inputSize;
            weightsData.HiddenSize1 = _hiddenSize1;
            weightsData.HiddenSize2 = _hiddenSize2;
            weightsData.OutputSize = _outputSize;

            // flatten _w1
            weightsData.W1 = new float[_hiddenSize1 * _inputSize];

            for (int i = 0; i < _hiddenSize1; i++)
            {
                for (int j = 0; j < _inputSize; j++)
                {
                    weightsData.W1[i * _inputSize + j] = _w1[i, j];
                }
            }

            // copy _b1
            weightsData.B1 = new float[_hiddenSize1];

            for (int i = 0; i < _hiddenSize1; i++)
            {
                weightsData.B1[i] = _b1[i];
            }

            // flatten _w2
            weightsData.W2 = new float[_hiddenSize2 * _hiddenSize1];

            for (int i = 0; i < _hiddenSize2; i++)
            {
                for (int j = 0; j < _hiddenSize1; j++)
                {
                    weightsData.W2[i * _hiddenSize1 + j] = _w2[i, j];
                }
            }

            // copy _b2
            weightsData.B2 = new float[_hiddenSize2];
            for (int i = 0; i < _hiddenSize2; i++)
            {
                weightsData.B2[i] = _b2[i];
            }

            // flatten _w3
            weightsData.W3 = new float[_outputSize * _hiddenSize2];

            for (int i = 0; i < _outputSize; i++)
            {
                for (int j = 0; j < _hiddenSize2; j++)
                {
                    weightsData.W3[i * _hiddenSize2 + j] = _w3[i, j];
                }
            }

            // copy _b3
            weightsData.B3 = new float[_outputSize];

            for (int i = 0; i < _outputSize; i++)
            {
                weightsData.B3[i] = _b3[i];
            }

            string json = JsonUtility.ToJson(weightsData, true);
            File.WriteAllText(filePath, json);
        }

        public void LoadWeights(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Debug.LogError($"No weights file located at: {filePath}");
                return;
            }

            string json = File.ReadAllText(filePath);
            NeuralNetworkWeightsData data = JsonUtility.FromJson<NeuralNetworkWeightsData>(json);

            for (int i = 0; i < _hiddenSize1; i++)
            {
                for (int j = 0; j < _inputSize; j++)
                {
                    _w1[i, j] = data.W1[i * _inputSize + j];
                }
            }

            for (int i = 0; i < _hiddenSize1; i++)
            {
                _b1[i] = data.B1[i];
            } 

            for (int i = 0; i < _hiddenSize2; i++)
            {
                for (int j = 0; j < _hiddenSize1; j++)
                {
                    _w2[i, j] = data.W2[i * _hiddenSize1 + j];
                }
            }

            for (int i = 0; i < _hiddenSize2; i++)
            {
                _b2[i] = data.B2[i];
            }

            for (int i = 0; i < _outputSize; i++)
            {
                for (int j = 0; j < _hiddenSize2; j++)
                {
                    _w3[i, j] = data.W3[i * _hiddenSize2 + j];
                }
            }

            for (int i = 0; i < _outputSize; i++)
            {
                _b3[i] = data.B3[i];
            }

            Debug.Log($"Weights loaded from: {filePath}");
        }
    }
}
