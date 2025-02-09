using System;
using TMPro;
using UnityEngine;

namespace SVGL
{
    public class UIManager : MonoBehaviour
    {
        [Header("UI Elements")]
        [SerializeField] private TextMeshProUGUI _resultElement;

        public static UIManager Instance { get; private set; }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this.gameObject);
                return;
            }

            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        private void Start()
        {
            _resultElement.text = string.Empty;
        }

        public void UpdateGuess(int number)
        {
            _resultElement.text = number.ToString();
        }

        public void ClearGuess()
        {
            if (_resultElement.text != string.Empty)
            {
                _resultElement.text = string.Empty;
            }
        }
    }
}
