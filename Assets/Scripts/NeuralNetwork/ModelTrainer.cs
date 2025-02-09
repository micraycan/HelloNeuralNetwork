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
        [SerializeField] private NetworkSettingsSO _settings;

        [Button("Train Model Weights")]
        public void TrainModel()
        {
#if UNITY_EDITOR
            NeuralNetwork neuralNetwork = new NeuralNetwork(_settings, true);

            string csvPath = Path.Combine(Application.dataPath, "StreamingAssets", _settings.TrainDataFile);
            csvPath = csvPath.Replace("\\", "/");

            if (!File.Exists(csvPath))
            {
                Debug.LogError("CSV file not found at path: " + csvPath);
                return;
            }

            string[] lines = File.ReadAllLines(csvPath);
            Debug.Log($"Total training samples in CSV: {lines.Length}");

            int examplesToTrain = Mathf.Min(_settings.TrainingSize, lines.Length);
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

                float[] pixels = new float[_settings.InputSize];
                for (int j = 0; j < _settings.InputSize; j++)
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

            string folderPath = Path.Combine(Application.dataPath, "StreamingAssets");
            folderPath = folderPath.Replace("\\", "/");

            string filePath = Path.Combine(folderPath, _settings.WeightDataFile);

            neuralNetwork.SaveWeights(filePath);

            Debug.Log($"Model weights exported to: {filePath}");
            ModelEvaluator.EvaluateTestSet(neuralNetwork, _settings);
#else
            Debug.LogError("How'd you even get here?");
#endif
        }
    }
}
