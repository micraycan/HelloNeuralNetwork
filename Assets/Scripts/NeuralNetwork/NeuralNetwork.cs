using System;
using System.IO;
using UnityEngine;
using Unity.Mathematics;
using System.Collections.Generic;

namespace SVGL
{
    [Serializable]
    public class NeuralNetwork
    {
        private float _learningRate;
        
        public Layer[] Layers { get; private set; }

        /// <summary>
        /// Initializes a new instance of the NeuralNetwork class using the specified settings.
        /// </summary>
        /// <param name="settings">The network configuration settings.</param>
        public NeuralNetwork(NetworkSettingsSO settings)
        {
            _learningRate = settings.LearningRate;

            Layers = new Layer[]
            {
                new Layer(settings.InputSize, settings.HiddenSize1, new Sigmoid()),
                new Layer(settings.HiddenSize1, settings.HiddenSize2, new Sigmoid()),
                new Layer(settings.HiddenSize2, settings.OutputSize, new Linear())
            };
        }

        /// <summary>
        /// Performs a forward pass through the network.
        /// </summary>
        /// <param name="inputs">The input values.</param>
        /// <returns>The softmax-normalized output probabilities.</returns>
        public float[] Forward(float[] inputs)
        {
            float[] current = inputs;

            foreach (Layer layer in Layers)
            {
                current = layer.Forward(current);
            }

            return Softmax(current);
        }

        /// <summary>
        /// Computes the softmax of an array of logits.
        /// </summary>
        /// <param name="logits">The logits to normalize.</param>
        /// <returns>An array of probabilities that sum to 1.</returns>
        private float[] Softmax(float[] logits)
        {
            float max = float.MinValue;
            float sum = 0f;
            float[] exp = new float[logits.Length];

            // find maximum logit (for numerical stability)
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > max) { max = logits[i]; }
            }

            // compute exponents and sum them
            for (int i = 0; i < logits.Length; i++)
            {
                exp[i] = math.exp(logits[i] - max);
                sum += exp[i];
            }

            // normalize exponentials to probabilities
            for (int i = 0; i < exp.Length; i++)
            {
                exp[i] /= sum;
            }

            return exp;
        }

        /// <summary>
        /// Trains the network on a single example using backpropagation.
        /// </summary>
        /// <param name="inputs">The input data.</param>
        /// <param name="target">The index of the correct target class.</param>
        public void Train(float[] inputs, int target)
        {
            // forward pass: record activations for each layer
            float[][] activations = new float[Layers.Length + 1][];
            activations[0] = inputs;
            float[] currentOutput = inputs;

            for (int i = 0; i < Layers.Length; i++)
            {
                currentOutput = Layers[i].Forward(currentOutput);
                activations[i + 1] = (float[])currentOutput.Clone();
            }

            float[] softmaxOutput = Softmax(currentOutput);
            float[] targetVector = new float[softmaxOutput.Length];

            // create one-hot encoded target vector
            for (int i = 0; i < targetVector.Length; i++)
            {
                targetVector[i] = (i == target) ? 1 : 0;
            }

            // compute initial error (softmax ouput minus target)
            float[] error = new float[softmaxOutput.Length];

            for (int i = 0; i < error.Length; i++)
            {
                error[i] = softmaxOutput[i] - targetVector[i];
            }

            // backpropogate the error through the layers
            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                error = Layers[i].Backward(activations[i], error, _learningRate);
            }
        }

        /// <summary>
        /// Saves the network weights to a JSON file.
        /// </summary>
        /// <param name="filePath">The file path where weights will be saved.</param>
        public void SaveWeights(string filePath)
        {
            NeuralNetworkData networkData = new NeuralNetworkData();
            networkData.Layers = new List<LayerData>();

            foreach (Layer layer in Layers)
            {
                LayerData layerData = new LayerData();

                layerData.NumNeurons = layer.Weights.GetLength(0);
                layerData.InputSize = layer.Weights.GetLength(1);

                layerData.Weights = new float[layerData.NumNeurons * layerData.InputSize];
                int index = 0;

                for (int i = 0; i < layerData.NumNeurons; i++)
                {
                    for (int j = 0; j < layerData.InputSize; j++)
                    {
                        layerData.Weights[index++] = layer.Weights[i, j];
                    }
                }

                layerData.Biases = new float[layerData.NumNeurons];

                for (int i = 0; i < layerData.NumNeurons; i++)
                {
                    layerData.Biases[i] = layer.Biases[i];
                }

                networkData.Layers.Add(layerData);
            }

            string json = JsonUtility.ToJson(networkData, true);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads network weights from a JSON file.
        /// </summary>
        /// <param name="filePath">The file path from which weights will be loaded.</param>
        public void LoadWeights(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Weights file not found: {filePath}");
            }

            string json = File.ReadAllText(filePath);
            LoadWeightsFromJSON(json);
        }

        /// <summary>
        /// Loads network weights from a JSON file.
        /// </summary>
        /// <param name="json">The json data to load.</param>
        public void LoadWeightsFromJSON(string json)
        {
            NeuralNetworkData networkData = JsonUtility.FromJson<NeuralNetworkData>(json);

            if (networkData.Layers.Count != Layers.Length)
            {
                throw new Exception("Mismatch with layers");
            }

            for (int l = 0; l < Layers.Length; l++)
            {
                LayerData layerData = networkData.Layers[l];
                int rows = layerData.NumNeurons;
                int cols = layerData.InputSize;

                if (rows != Layers[l].Weights.GetLength(0) || cols != Layers[l].Weights.GetLength(1))
                {
                    throw new Exception("Dimension mismatch");
                }

                int index = 0;

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        Layers[l].Weights[i, j] = layerData.Weights[index++];
                    }
                }

                Array.Copy(layerData.Biases, Layers[l].Biases, rows);
            }
        }
    }
}
