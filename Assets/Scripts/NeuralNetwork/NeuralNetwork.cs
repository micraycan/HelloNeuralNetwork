using System;
using System.IO;
using UnityEngine;

namespace SVGL
{
    [Serializable]
    public class NeuralNetwork
    {
        public readonly int INPUT_SIZE = 784;
        public readonly int HIDDEN_SIZE = 128;
        public readonly int OUTPUT_SIZE = 10;
        public readonly float LEARNING_RATE = 0.05f;

        public float[,] W1;
        public float[] B1;
        public float[,] W2;
        public float[] B2; 

        System.Random rand = new System.Random();

        public NeuralNetwork()
        {
            W1 = new float[HIDDEN_SIZE, INPUT_SIZE];
            B1 = new float[HIDDEN_SIZE];
            W2 = new float[OUTPUT_SIZE, HIDDEN_SIZE];
            B2 = new float[OUTPUT_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                for (int j = 0; j < INPUT_SIZE; j++)
                {
                    W1[i, j] = (float)(rand.NextDouble() * 0.1 - 0.05);
                }

                B1[i] = 0;
            }

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                for (int j = 0; j < HIDDEN_SIZE; j++)
                {
                    W2[i, j] = (float)(rand.NextDouble() * 0.1 - 0.05);
                }
                B2[i] = 0;
            }
        }

        public float[] Forward(float[] input, out float[] hidden, out float[] logits)
        {
            hidden = new float[HIDDEN_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                float sum = B1[i];
                for (int j = 0; j < INPUT_SIZE; j++)
                {
                    sum += W1[i, j] * input[j];
                }

                hidden[i] = Utils.Sigmoid(sum);
            }

            logits = new float[OUTPUT_SIZE];

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                float sum = B2[i];

                for (int j = 0; j < HIDDEN_SIZE; j++)
                {
                    sum += W2[i, j] * hidden[j];
                }

                logits[i] = sum;
            }

            float[] output = Utils.Softmax(logits);
            return output;
        }

        public void TrainOnSample(float[] input, int label)
        {
            float[] hidden, logits;
            float[] output = Forward(input, out hidden, out logits);

            float[] target = new float[OUTPUT_SIZE];
            target[label] = 1;

            // compute error at output
            float[] errorOutput = new float[OUTPUT_SIZE];
            
            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                errorOutput[i] = output[i] - target[i];
            }

            // W2 and B2 gradients
            float[,] gradientW2 = new float[OUTPUT_SIZE, HIDDEN_SIZE];
            float[] gradientB2 = new float[OUTPUT_SIZE];

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                gradientB2[i] = errorOutput[i];

                for (int j = 0; j < HIDDEN_SIZE; j++)
                {
                    gradientW2[i, j] = errorOutput[i] * hidden[j];
                }
            }

            // backpropogate error to hidden layer
            float[] errorHidden = new float[HIDDEN_SIZE];

            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                float sum = 0;

                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    sum += W2[i, j] * errorOutput[i];
                }

                errorHidden[j] = sum * Utils.SigmoidDerivative(hidden[j]);
            }

            // W1 and B1 gradients
            float[,] gradientW1 = new float[HIDDEN_SIZE, INPUT_SIZE];
            float[] gradientB1 = new float[HIDDEN_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                gradientB1[i] = errorHidden[i];
                for (int j = 0; j < INPUT_SIZE; j++)
                {
                    gradientW1[i, j] = errorHidden[i] * input[j];
                }
            }

            // update weights
            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                for (int j = 0; j < INPUT_SIZE; j++)
                {
                    W1[i, j] -= LEARNING_RATE * gradientW1[i, j];
                }

                B1[i] -= LEARNING_RATE * gradientB1[i];
            }

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                for (int j = 0; j < HIDDEN_SIZE; j++)
                {
                    W2[i, j] -= LEARNING_RATE * gradientW2[i, j];
                }

                B2[i] -= LEARNING_RATE * gradientB2[i];
            }
        }

        public void SaveWeights(string filePath)
        {
            NeuralNetworkWeightsData weightsData = new NeuralNetworkWeightsData();
            weightsData.InputSize = INPUT_SIZE;
            weightsData.HiddenSize = HIDDEN_SIZE;
            weightsData.OutputSize = OUTPUT_SIZE;

            // flatten W1
            weightsData.W1 = new float[HIDDEN_SIZE * INPUT_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                for (int j = 0; j < INPUT_SIZE; j++)
                {
                    weightsData.W1[i * INPUT_SIZE + j] = W1[i, j];
                }
            }

            // copy B1
            weightsData.B1 = new float[HIDDEN_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                weightsData.B1[i] = B1[i];
            }

            // flatten W2
            weightsData.W2 = new float[OUTPUT_SIZE * HIDDEN_SIZE];

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                for (int j = 0; j < HIDDEN_SIZE; j++)
                {
                    weightsData.W2[i * HIDDEN_SIZE + j] = W2[i, j];
                }
            }

            // copy B2
            weightsData.B2 = new float[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                weightsData.B2[i] = B2[i];
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

            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                for (int j = 0; j < INPUT_SIZE; j++)
                {
                    W1[i, j] = data.W1[i * INPUT_SIZE + j];
                }
            }

            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                B1[i] = data.B1[i];
            } 

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                for (int j = 0; j < HIDDEN_SIZE; j++)
                {
                    W2[i, j] = data.W2[i * HIDDEN_SIZE + j];
                }
            }

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                B2[i] = data.B2[i];
            }

            Debug.Log($"Weights loaded from: {filePath}");
            ModelEvaluator.EvaluateTestSet(this);
        }
    }
}
