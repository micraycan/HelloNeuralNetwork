using System.Globalization;
using System.IO;
using UnityEngine;

namespace SVGL
{
    public static class ModelEvaluator
    {
        /// <summary>
        /// Evaluates the given neural network using the test data specified in the settings.
        /// Reads a CSV file containing test samples, processes each sample, and logs the overall accuracy.
        /// </summary>
        /// <param name="neuralNetwork">The trained neural network to be evaluated.</param>
        /// <param name="settings">The network settings, including file paths for test data.</param>
        public static void EvaluateTestSet(NeuralNetwork neuralNetwork, NetworkSettingsSO settings)
        {
            string testFilePath = Path.Combine(Application.streamingAssetsPath, settings.TestDataFile);

            if (!File.Exists(testFilePath))
            {
                throw new FileNotFoundException($"Test file not found: {testFilePath}");
            }

            string[] lines = File.ReadAllLines(testFilePath);
            int totalSamples = lines.Length;
            int correctCount = 0;

            for (int i = 0; i < totalSamples; i++)
            {
                string line = lines[i];
                string[] parts = line.Split(',');

                if (parts.Length != 785)
                {
                    continue;
                }

                if (!int.TryParse(parts[0], out int label))
                {
                    continue;
                }

                float[] pixels = new float[784];
                for (int j = 0; j < 784; j++)
                {
                    if (!float.TryParse(parts[j + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float pixelValue))
                    {
                        pixelValue = 0;
                    }

                    pixels[j] = pixelValue / 255;
                }

                // get neural network output for the current sample
                float[] output = neuralNetwork.Forward(pixels);

                // determine the predicted label (index of maximum probability).
                int predictedLabel = 0;
                float maxProbability = output[0];
                for (int k = 1; k < output.Length; k++)
                {
                    if (output[k] > maxProbability)
                    {
                        maxProbability = output[k];
                        predictedLabel = k;
                    }
                }

                if (predictedLabel == label)
                {
                    correctCount++;
                }
            }

            float accuracy = (float)correctCount / totalSamples * 100f;
        }
    }
}
