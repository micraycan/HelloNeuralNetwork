using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace SVGL
{
    public class DrawingCanvas : MonoBehaviour, IPointerDownHandler, IDragHandler
    {
        [SerializeField] private int _brushSize;
        private Vector2? _lastDrawPosition = null;

        private RawImage _canvas;
        private Texture2D _image;
        private Vector2 _size;

        private void Start()
        {
            _canvas = GetComponent<RawImage>();
            _size = _canvas.rectTransform.sizeDelta;
            _image = new Texture2D((int)_size.x, (int)_size.y);
            _canvas.texture = _image;

            ResetDrawing();
        }

        public void OnDrag(PointerEventData eventData) => Draw(eventData);

        public void OnPointerDown(PointerEventData eventData) => _lastDrawPosition = null;

        private bool CanDraw(RectTransform rect, PointerEventData eventData, out Vector2 localPos)
        {
            return RectTransformUtility.ScreenPointToLocalPointInRectangle(rect, eventData.position, eventData.pressEventCamera, out localPos);
        }

        private void Draw(PointerEventData eventData)
        {
            if (CanDraw(_canvas.rectTransform, eventData, out Vector2 localPos))
            {
                UIManager.Instance.ClearGuess();

                float x = localPos.x + _canvas.rectTransform.rect.width * _canvas.rectTransform.pivot.x;
                float y = localPos.y + _canvas.rectTransform.rect.height * _canvas.rectTransform.pivot.y;

                Vector2 currentPos = new Vector2(x, y);

                if (_lastDrawPosition.HasValue) { DrawLine(_lastDrawPosition.Value, currentPos); }
                else { DrawWithBrush((int)x, (int)y); }

                _lastDrawPosition = currentPos;
            }
        }

        private void DrawLine(Vector2 start, Vector2 end)
        {
            float distance = Vector2.Distance(start, end);
            int steps = Mathf.CeilToInt(distance);

            for (int i = 0; i <= steps; i++)
            {
                float time = i / (float)steps;
                Vector2 interpolatedPoint = Vector2.Lerp(start, end, time);
                DrawWithBrush((int)interpolatedPoint.x, (int)interpolatedPoint.y);
            }
        }

        private void DrawWithBrush(int x, int y)
        {
            int brushRadius = _brushSize / 2;

            for (int i = -brushRadius; i < brushRadius / 2; i++)
            {
                for (int j = -brushRadius; j < brushRadius / 2; j++)
                {
                    int pixelX = x + i;
                    int pixelY = y + j;
                    float distance = Mathf.Sqrt(i * i + j * j);

                    if (distance <= brushRadius)
                    {
                        _image.SetPixel(pixelX, pixelY, Color.white);
                    }
                }
            }

            _image.Apply();
        }

        public void ResetDrawing()
        {
            Color[] clearColors = new Color[(int)(_size.x * _size.y)];
            for (int i = 0; i < clearColors.Length; i++) { clearColors[i] = Color.black; }
            _image.SetPixels(clearColors);
            _image.Apply();
        }
    }
}
