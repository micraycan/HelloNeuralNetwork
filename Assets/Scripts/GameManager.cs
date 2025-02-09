using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

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

            
        }

        private void Start()
        {
            string filePath = string.Empty;
#if UNITY_WEBGL
            filePath = Application.streamingAssetsPath + "/" + NetworkSettings.WeightDataFile;
            StartCoroutine(LoadWeightsWebGL(filePath));
#else
            filePath = Path.Combine(Application.streamingAssetsPath, NetworkSettings.WeightDataFile);
            NeuralNet.LoadWeights(filePath); }
#endif
        }

        private IEnumerator LoadWeightsWebGL(string filePath)
        {
            using (UnityWebRequest webRq = UnityWebRequest.Get(filePath))
            {
                yield return webRq.SendWebRequest();

                switch (webRq.result)
                {
                    case UnityWebRequest.Result.ConnectionError:
                        throw new IOException("Connection Error");
                    case UnityWebRequest.Result.ProtocolError:
                        throw new IOException("Protocol Error");
                    default:
                        string jsonText = webRq.downloadHandler.text;
                        NeuralNet.LoadWeightsFromJSON(jsonText);
                        break;
                }
            }
        }
    }
}
