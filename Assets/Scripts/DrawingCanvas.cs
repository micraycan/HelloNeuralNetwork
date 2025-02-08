using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace SVGL
{
    public class DrawingCanvas : MonoBehaviour, IPointerDownHandler, IDragHandler
    {
        [SerializeField] private int _brushSize;

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

        public void OnDrag(PointerEventData eventData)
        {
            Draw(eventData);
        }

        public void OnPointerDown(PointerEventData eventData)
        {
            //Draw(eventData);
        }

        private bool CanDraw(RectTransform rect, PointerEventData eventData, out Vector2 localPos)
        {
            return RectTransformUtility.ScreenPointToLocalPointInRectangle(rect, eventData.position, eventData.pressEventCamera, out localPos);
        }

        private void Draw(PointerEventData eventData)
        {
            if (CanDraw(_canvas.rectTransform, eventData, out Vector2 localPos))
            {
                float x = localPos.x + _canvas.rectTransform.rect.width * _canvas.rectTransform.pivot.x;
                float y = localPos.y + _canvas.rectTransform.rect.height * _canvas.rectTransform.pivot.y;

                DrawWithBrush((int)x, (int)y);
            }
        }

        private void DrawWithBrush(int x, int y)
        {
            for (int i = -_brushSize / 2; i < _brushSize / 2; i++)
            {
                for (int j = -_brushSize / 2; j < _brushSize / 2; j++)
                {
                    _image.SetPixel(x + i, y + j, Color.white);
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
