using System;
using System.Collections.Generic;

namespace SVGL
{
    [Serializable]
    public class LayerData 
    {
        public int NumNeurons;
        public int InputSize;
        public float[] Weights;
        public float[] Biases;
    }
}
