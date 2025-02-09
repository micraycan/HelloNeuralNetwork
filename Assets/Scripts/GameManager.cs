using System.IO;
using UnityEngine;

namespace SVGL
{
    public class GameManager : MonoBehaviour
    {
        public static GameManager Instance { get; private set; }
        public NeuralNetwork NeuralNet { get; private set; }
        [field: SerializeField] public NetworkSettingsSO NetworkSettings { get; private set; }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this.gameObject);
                return;
            }

            Instance = this;
            DontDestroyOnLoad(gameObject);

            NeuralNet = new NeuralNetwork(NetworkSettings);

            string filePath = Path.Combine(Application.dataPath, "StreamingAssets", NetworkSettings.WeightDataFile);
            filePath = filePath.Replace("\\", "/");
            NeuralNet.LoadWeights(filePath);
        }
    }
}
