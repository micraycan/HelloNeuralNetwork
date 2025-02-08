using UnityEngine;

namespace SVGL
{
    public class ImageProcessor
    {
        private Texture2D _image;

        public ImageProcessor(Texture2D image)
        {
            _image = image;
        }

        public float[] ImagePixelData
        {
            get
            {
                var scaledImage = ResizeImage(_image, 28, 28);
                Color[] pixels = scaledImage.GetPixels();
                float[] pixelData = new float[pixels.Length];

                int width = scaledImage.width;
                int height = scaledImage.height;

                for (int y = height - 1; y >= 0; y--)
                {
                    int row = height - 1 - y;

                    for (int x = 0; x < width; x++)
                    {
                        int index = row * width + x;
                        pixelData[index] = scaledImage.GetPixel(x, y).grayscale;
                    }
                }

                return pixelData;
            }
        }

        public Texture2D ResizedImage
        {
            get
            {
                return ResizeImage(_image, 28, 28);
            }
        }

        private Texture2D ResizeImage(Texture2D image, int targetWidth, int targetHeight)
        {
            RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight, 0, RenderTextureFormat.Default, RenderTextureReadWrite.Linear);
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = rt;

            Graphics.Blit(image, rt);

            Texture2D result = new Texture2D(targetWidth, targetHeight, image.format, false);

            result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
            result.Apply();

            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(rt);

            return result;
        }
    }
}
