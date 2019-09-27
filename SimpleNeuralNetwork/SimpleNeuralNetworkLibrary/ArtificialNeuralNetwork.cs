using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkLibrary
{
    public class ArtificialNeuralNetwork
    {
        public List<Layer> Layers { get; } = new List<Layer>();

        public ArtificialNeuralNetwork(ArtificialNeuralNetworkConfig config)
        {
            foreach (int neuronCount in config.NeuronCounts)
            {
                var prevLayer = Layers.Count > 0 ? Layers[Layers.Count - 1] : null;
                Layers.Add(new Layer(neuronCount, prevLayer));
            }

            Layers[0].CreateInputConnections(config.InputDimensions);
            Layers[Layers.Count - 1].CreateOutputConnections();
        }

        /// <summary>
        /// Forward-propagate a data point through the ANN.
        /// </summary>
        /// <param name="input">Input values, not including bias. The number of dimensions must match the configured ANN.</param>
        /// <returns>The output from the ANN.</returns>
        public double[] Classify(double[] input)
        {
            foreach (var neuron in Layers[0].Neurons)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    // Put the value of each input value to the incoming connections of all the layer's neurons
                    // Note: The first incoming connection to a neuron is the bias/threshold, which is not changed.
                    neuron.IncomingConnections[i + 1].Activation = input[i];
                }
            }

            // Forward-propagate the values through the network
            foreach (Layer layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    var activationValue = neuron.Activation;

                    foreach (Connection outgoingConnection in neuron.OutgoingConnections)
                    {
                        outgoingConnection.Activation = activationValue;
                    }
                }
            }

            // Collect values from output layer
            return Layers[Layers.Count - 1].AllOutputs();
        }
    }
}
