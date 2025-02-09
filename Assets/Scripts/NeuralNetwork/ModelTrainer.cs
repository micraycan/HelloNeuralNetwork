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

        [Button("Train Model Weights")]
        public void TrainModel()
        {
#if UNITY_EDITOR
            NeuralNetwork neuralNet = new NeuralNetwork(_settings);
            string csvPath = Path.Combine(Application.streamingAssetsPath, _settings.TrainDataFile);
            string[] lines = File.ReadAllLines(csvPath);

            for (int epoch = 0; epoch < _settings.Epochs; epoch++)
            {
                string[] shuffledLines = lines.OrderBy(x => Guid.NewGuid()).ToArray();

                foreach (string line in shuffledLines)
                {
                    ProcessSample(neuralNet, line);
                }

                ModelEvaluator.EvaluateTestSet(neuralNet, _settings);
            }

            string filePath = Path.Combine(Application.streamingAssetsPath, _settings.WeightDataFile);
            neuralNet.SaveWeights(filePath);
#else
            Debug.LogError("How'd you even get here?");
#endif
        }

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

                pixels[j] = Mathf.Clamp(value / 255, 0, 1);
            }

            neuralNet.Train(pixels, label);
        }
    }
}
