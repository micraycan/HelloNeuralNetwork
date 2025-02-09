namespace SVGL
{
    public interface IActivationFunction
    {
        float Activate(float x);
        float Derivative(float y);
    }
}
