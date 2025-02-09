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

        public float[] Forward(float[] inputs)
        {
            float[] current = inputs;

            foreach (Layer layer in Layers)
            {
                current = layer.Forward(current);
            }

            return Softmax(current);
        }

        private float[] Softmax(float[] logits)
        {
            float max = float.MinValue;
            float sum = 0f;
            float[] exp = new float[logits.Length];

            // find max value
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > max) { max = logits[i]; }
            }

            // compute exponents
            for (int i = 0; i < logits.Length; i++)
            {
                exp[i] = math.exp(logits[i] - max);
                sum += exp[i];
            }

            // normalize probabilities
            for (int i = 0; i < exp.Length; i++)
            {
                exp[i] /= sum;
            }

            return exp;
        }

        public void Train(float[] inputs, int target)
        {
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

            for (int i = 0; i < targetVector.Length; i++)
            {
                targetVector[i] = (i == target) ? 1 : 0;
            }

            float[] error = new float[softmaxOutput.Length];

            for (int i = 0; i < error.Length; i++)
            {
                error[i] = softmaxOutput[i] - targetVector[i];
            }

            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                error = Layers[i].Backward(activations[i], error, _learningRate);
            }
        }

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
            Debug.Log($"Weights saved to: {filePath}");
        }

        public void LoadWeights(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Weights file not found: {filePath}");
            }

            string json = File.ReadAllText(filePath);
            NeuralNetworkData networkData = JsonUtility.FromJson<NeuralNetworkData>(json);

            if (networkData.Layers.Count != Layers.Length)
            {
                Debug.LogError($"Mismatch in number of layers. Expected: {Layers.Length}; Found: {networkData.Layers.Count}");
                return;
            }

            for (int l = 0; l < Layers.Length; l++)
            {
                LayerData layerData = networkData.Layers[l];
                int rows = layerData.NumNeurons;
                int cols = layerData.InputSize;

                if (rows != Layers[l].Weights.GetLength(0) || cols != Layers[l].Weights.GetLength(1))
                {
                    Debug.LogError($"Dimension mismatch for layer {l}");
                    continue;
                }

                int index = 0;

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        Layers[l].Weights[i, j] = layerData.Weights[index++];
                    }
                }

                for (int i = 0; i < rows; i++)
                {
                    Layers[l].Biases[i] = layerData.Biases[i];
                }
            }

            Debug.Log($"Weights loaded from: {filePath}");
        }
    }
}
