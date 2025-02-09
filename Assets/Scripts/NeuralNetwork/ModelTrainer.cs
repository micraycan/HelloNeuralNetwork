using UnityEngine;
using System.IO;
using Sirenix.OdinInspector;
using Unity.Mathematics;
using System.Linq;
using System;



#if UNITY_EDITOR
#endif
using System.Globalization;

namespace SVGL
{
    public class ModelTrainer : MonoBehaviour
    {
        [SerializeField] private NetworkSettingsSO _settings;

        /// <summary>
        /// Trains the neural network model using training data from a CSV file.
        /// This method is triggered via an Odin Inspector button.
        /// </summary>
        [Button("Train Model Weights")]
        public void TrainModel()
        {
#if UNITY_EDITOR
            NeuralNetwork neuralNet = new NeuralNetwork(_settings);
            string csvPath = Path.Combine(Application.streamingAssetsPath, _settings.TrainDataFile);
            string[] lines = File.ReadAllLines(csvPath);

            for (int epoch = 0; epoch < _settings.Epochs; epoch++)
            {
                // shuffle the dataset by ordering randomly
                string[] shuffledLines = lines.OrderBy(x => Guid.NewGuid()).ToArray();

                foreach (string line in shuffledLines)
                {
                    ProcessSample(neuralNet, line);
                }

                // evaluate the neural network on a test set after each epoch
                ModelEvaluator.EvaluateTestSet(neuralNet, _settings);
            }

            string filePath = Path.Combine(Application.streamingAssetsPath, _settings.WeightDataFile);
            neuralNet.SaveWeights(filePath);
#else
            Debug.LogError("How'd you even get here?");
#endif
        }

        /// <summary>
        /// Processes a single CSV sample by extracting the label and pixel data,
        /// normalizing the pixel values, and training the neural network with the sample.
        /// </summary>
        /// <param name="neuralNet">The neural network to be trained.</param>
        /// <param name="line">A line from the CSV file containing the label and pixel values.</param>
        private void ProcessSample(NeuralNetwork neuralNet, string line)
        {
            string[] parts = line.Split(',');
            int label = int.Parse(parts[0]);
            float[] pixels = new float[_settings.InputSize];

            for (int j = 0; j < pixels.Length; j++)
            {
                if (!float.TryParse(parts[j + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float value))
                {
                    value = 0f;
                }

                // normalize the pixel value from [0, 255] to a [0, 1] range
                pixels[j] = Mathf.Clamp(value / 255, 0, 1);
            }

            neuralNet.Train(pixels, label);
        }
    }
}
