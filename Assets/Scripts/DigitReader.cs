using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace SVGL
{
    public class DigitReader : MonoBehaviour
    {
        private NeuralNetwork _neuralNet;

        [SerializeField] private RawImage _drawCanvas;
        [SerializeField] private RawImage _scaledResult;
        [SerializeField] private string _trainedData;

        private void Start()
        {
            string weightsPath = Path.Combine(Application.dataPath, "StreamingAssets", _trainedData);
            weightsPath = weightsPath.Replace("\\", "/");

            _neuralNet = new NeuralNetwork();
            _neuralNet.LoadWeights(weightsPath);
        }

        public void Execute()
        {
            Texture2D image = (Texture2D)_drawCanvas.texture;
            ImageProcessor processor = new ImageProcessor(image);

            _scaledResult.texture = processor.ResizedImage;

            float[] result = _neuralNet.Forward(processor.ImagePixelData, out float[] hidden, out float[] logits);

            int highestIndex = 0;
            float highestWeight = float.NegativeInfinity;

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] > highestWeight)
                {
                    highestIndex = i;
                    highestWeight = result[i];
                }
            }

            Debug.Log($"Guess: {highestIndex} | Confidence: {highestWeight}");
        }
    }
}
