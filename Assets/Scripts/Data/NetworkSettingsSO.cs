using UnityEditor;
using UnityEngine;

namespace SVGL
{
    [CreateAssetMenu(fileName = "Network Settings", menuName = "Data/Neueral Network Settings")]
    public class NetworkSettingsSO : ScriptableObject
    {
        [Header("Files")]
        [SerializeField] private string trainCSV;
        [SerializeField] private string testCSV;
        [SerializeField] private string weightsJSON;

        [Header("Neural Network Settings")]
        [SerializeField] private int _inputSize;
        [SerializeField] private int _hiddenSize1;
        [SerializeField] private int _hiddenSize2;
        [SerializeField] private int _outputSize;

        [Header("Training Parameters")]
        [SerializeField] private float _learningRate;
        [SerializeField] private int _epochs;

        public string TrainDataFile => trainCSV;
        public string TestDataFile => testCSV;
        public string WeightDataFile => weightsJSON;

        public int InputSize => _inputSize;
        public int HiddenSize1 => _hiddenSize1;
        public int HiddenSize2 => _hiddenSize2;
        public int OutputSize => _outputSize;

        public float LearningRate => _learningRate;
        public int Epochs => _epochs;
    }
}
