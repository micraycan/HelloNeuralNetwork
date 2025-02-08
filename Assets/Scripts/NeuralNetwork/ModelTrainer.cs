using UnityEngine;
using System.IO;
using Sirenix.OdinInspector;

#if UNITY_EDITOR
using UnityEditor;
#endif
using System.Globalization;

namespace SVGL
{
    public class ModelTrainer : MonoBehaviour
    {
        [SerializeField] private int _trainingCount = 1000;
        [SerializeField] private string _trainingData;
        [SerializeField] private string _testingData;
        [SerializeField] private string _trainedData;

        [Button("Train Model Weights")]
        public void TrainModel()
        {
#if UNITY_EDITOR
            NeuralNetwork neuralNetwork = new NeuralNetwork();
            var time = Time.time;

            string csvPath = Path.Combine(Application.dataPath, "StreamingAssets", _trainingData);
            csvPath = csvPath.Replace("\\", "/");

            if (!File.Exists(csvPath))
            {
                Debug.LogError("CSV file not found at path: " + csvPath);
                return;
            }

            string[] lines = File.ReadAllLines(csvPath);
            Debug.Log($"Total training samples in CSV: {lines.Length}");

            int examplesToTrain = Mathf.Min(_trainingCount, lines.Length);
            Debug.Log($"Training on {examplesToTrain} samples...");

            for (int i = 0; i < examplesToTrain; i++)
            {
                string line = lines[i];
                string[] parts = line.Split(',');

                if (parts.Length != 785) // 784 pixels + 1 label
                {
                    Debug.LogWarning($"Skipping invalid line {i}");
                    continue;
                }

                if (!int.TryParse(parts[0], out int label))
                {
                    Debug.LogWarning($"Skipping invalid label {parts[0]}");
                    continue;
                }

                float[] pixels = new float[neuralNetwork.INPUT_SIZE];
                for (int j = 0; j < neuralNetwork.INPUT_SIZE; j++)
                {
                    if (!float.TryParse(parts[j + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float pixelValue))
                    {
                        Debug.LogWarning($"Skipping invalid pixel {parts[j + 1]}");
                        pixelValue = 0;
                        continue;
                    }

                    pixels[j] = pixelValue / 255f; // Normalize pixel values
                }

                neuralNetwork.TrainOnSample(pixels, label);

                if (i % 1000 == 0) { Debug.Log($"Trained on {i} samples..."); }
            }

            var duration = Time.time - time;
            Debug.Log($"Training completed in {duration:F2} seconds ");

            string folderPath = Path.Combine(Application.dataPath, "StreamingAssets");
            folderPath = folderPath.Replace("\\", "/");

            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            string filePath = Path.Combine(folderPath, _trainedData);

            neuralNetwork.SaveWeights(filePath);

            Debug.Log($"Model weights exported to: {filePath}");
#else
            Debug.LogError("How'd you even get here?");
#endif
        }
    }
}
