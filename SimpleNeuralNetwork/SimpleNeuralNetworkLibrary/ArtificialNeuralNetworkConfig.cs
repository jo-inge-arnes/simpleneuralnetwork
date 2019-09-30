namespace SimpleNeuralNetworkLibrary
{
    public class ArtificialNeuralNetworkConfig
    {
        public ActivationTypeEnum ActivationType { get; set; }

        /// <summary>
        /// The number of dimensions in the input data
        /// </summary>
        public int InputDimensions { get; set; }

        /// <summary>
        /// Step size for gradient descent/backpropagation
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// An array with one entry per layer with the
        /// wanted number of neurons for the layer.
        /// Note that the number of neurons in the
        /// last layer defines the number of dimensions
        /// in the output.
        /// </summary>
        public int[] NeuronCounts { get; set; } 
    }

    public enum ActivationTypeEnum
    {
        NotSet,
        ReLU,
        Sigmoid
    }
}