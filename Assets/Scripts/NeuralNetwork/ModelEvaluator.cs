using System.Globalization;
using System.IO;
using UnityEngine;

namespace SVGL
{
    public static class ModelEvaluator
    {
        public static void EvaluateTestSet(NeuralNetwork neuralNetwork, NetworkSettingsSO settings)
        {
            string testFilePath = Path.Combine(Application.streamingAssetsPath, settings.TestDataFile);
            testFilePath = testFilePath.Replace("\\", "/");

            if (!File.Exists(testFilePath))
            {
                Debug.LogError($"Test file not found: {testFilePath}");
                return;
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
                    Debug.LogWarning($"Skipping invalid line {i}");
                    continue;
                }

                if (!int.TryParse(parts[0], out int label))
                {
                    Debug.LogWarning($"Skipping invalid label at line {i}");
                    continue;
                }

                float[] pixels = new float[784];
                for (int j = 0; j < 784; j++)
                {
                    if (!float.TryParse(parts[j + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float pixelValue))
                    {
                        Debug.LogWarning($"Skipping invalid pixel value at line {i}, index {j}");
                        pixelValue = 0;
                    }

                    pixels[j] = pixelValue / 255;
                }

                float[] output = neuralNetwork.Forward(pixels);

                // Determine the predicted label (index of maximum probability).
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
            Debug.Log($"Test Accuracy: {accuracy:F2}%");
        }
    }
}
