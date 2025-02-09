using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace SVGL
{
    public class DigitReader : MonoBehaviour
    {
        [SerializeField] private RawImage _drawCanvas;

        public void Execute()
        {
            Texture2D image = (Texture2D)_drawCanvas.texture;
            ImageProcessor processor = new ImageProcessor(image);
            NeuralNetwork neuralNet = GameManager.Instance.NeuralNet;

            float[] result = neuralNet.Forward(processor.ImagePixelData);

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

            _drawCanvas.GetComponent<DrawingCanvas>().ResetDrawing();
            UIManager.Instance.UpdateGuess(highestIndex);
        }
    }
}
