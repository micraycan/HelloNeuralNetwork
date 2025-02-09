using UnityEditor;
using UnityEngine;

namespace SVGL
{
    [CreateAssetMenu(fileName = "Network Settings", menuName = "Data/Neueral Network Settings")]
    public class NetworkSettingsSO : ScriptableObject
    {
        [Header("Files")]
        [SerializeField] private DefaultAsset trainCSV;
        [SerializeField] private DefaultAsset testCSV;
        [SerializeField] private DefaultAsset weightsJSON;

        [Header("Neural Network Settings")]
        [field: SerializeField] public int InputSize { get; set; }
        [field: SerializeField] public int HiddenSize1 { get; set; }
        [field: SerializeField] public int HiddenSize2 { get; set; }
        [field: SerializeField] public int OutputSize { get; set; }

        [Header("Training Parameters")]
        [field: SerializeField] public int TrainingSize { get; set; }
        [field: SerializeField] public float LearningRate { get; set; }

        public string TrainDataFile => trainCSV.name + ".csv";
        public string TestDataFile => testCSV.name + ".csv";
        public string WeightDataFile => weightsJSON.name + ".json";
    }
}
