using Unity.Mathematics;

namespace SVGL
{
    public class Sigmoid : IActivationFunction
    {
        public float Activate(float x) => 1f / (1f + math.exp(-x));
        public float Derivative(float y) => y * (1f - y);
    }
}
