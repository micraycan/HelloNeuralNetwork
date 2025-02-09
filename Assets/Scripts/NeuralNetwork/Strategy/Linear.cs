using Unity.Mathematics;

namespace SVGL
{
    public class Linear : IActivationFunction
    {
        public float Activate(float x) => x;
        public float Derivative(float y) => 1f;
    }
}
