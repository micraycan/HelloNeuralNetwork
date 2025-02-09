

using Unity.Mathematics;

namespace SVGL
{
    public class ReLU : IActivationFunction
    {
        public float Activate(float x) => math.max(0f, x);
        public float Derivative(float y) => y > 0f ? 1f : 0f;
    }
}
