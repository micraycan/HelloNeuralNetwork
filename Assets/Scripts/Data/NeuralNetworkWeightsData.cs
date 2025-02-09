using System;
using UnityEngine;

namespace SVGL
{
    [Serializable]
    public class NeuralNetworkWeightsData 
    {
        public int InputSize;
        public int HiddenSize1;
        public int HiddenSize2;
        public int OutputSize;

        public float[] W1;
        public float[] B1;
        public float[] W2;
        public float[] B2;
        public float[] W3;
        public float[] B3;
    }
}
