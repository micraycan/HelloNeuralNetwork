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
        [SerializeField] private int _inputSize;
        [SerializeField] private int _hiddenSize1;
        [SerializeField] private int _hiddenSize2;
        [SerializeField] private int _outputSize;

        [Header("Training Parameters")]
        [SerializeField] private int _trainingSize;
        [SerializeField] private float _learningRate;
        [SerializeField] private int _epoch;

        public string TrainDataFile => trainCSV.name + ".csv";
        public string TestDataFile => testCSV.name + ".csv";
        public string WeightDataFile => weightsJSON.name + ".json";

        public int InputSize => _inputSize;
        public int HiddenSize1 => _hiddenSize1;
        public int HiddenSize2 => _hiddenSize2;
        public int OutputSize => _outputSize;

        public int TrainingSize => _trainingSize;
        public float LearningRate => _learningRate;
        public int Epoch => _epoch;
    }
}
