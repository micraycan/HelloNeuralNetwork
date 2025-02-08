using UnityEngine;

namespace SVGL
{
    public static class Utils
    {
        /// <summary>
        /// Applies sigmoid activation function to given input.
        /// Sigmoid function maps any input to a value between 0 and 1.
        /// Formula: f(x) = 1 / (1 + e^(-x))
        /// </summary>
        /// <param name="x">Input value to the sigmoid function.</param>
        /// <returns>Sigmoid of the input value, ranging between 0 and 1</returns>
        public static float Sigmoid(float x)
        {
            return 1f / (1f + Mathf.Exp(-x));
        }

        /// <summary>
        /// Calculates the derivative of the sigmoid function given its output value.
        /// Used for backpropagation for updating weights.
        /// Formula: f'(x) = f(x) * (1 - f(x))
        /// </summary>
        /// <param name="sigmoidValue">Output value of the sigmoid function.</param>
        /// <returns>Derivative of the sigmoid function.</returns>
        public static float SigmoidDerivative(float sigmoidValue)
        {
            return sigmoidValue * (1 - sigmoidValue);
        }

        /// <summary>
        /// Applies the softmax function to an array of input values.
        /// Softmax function converts a vector of values into a probability distribution, where the sum of all outputs is 1.
        /// Formula: softmax(xi) = exp(xi - max(x)) / sum(exp(xj - max(x))
        /// This prevents numerical instability by subtracting the max value.
        /// </summary>
        /// <param name="x">Array of input values.</param>
        /// <returns>Array of values representing the softmax probabilities.</returns>
        public static float[] Softmax(float[] x)
        {
            float max = x[0];

            for (int i = 1; i < x.Length; i++)
            {
                if (x[i] > max) max = x[i];
            }

            float sum = 0f;
            float[] exp = new float[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = Mathf.Exp(x[i] - max);
                sum += exp[i];
            }

            for (int i = 0; i < x.Length; i++)
            {
                exp[i] /= sum;
            }

            return exp;
        }
    }
}
