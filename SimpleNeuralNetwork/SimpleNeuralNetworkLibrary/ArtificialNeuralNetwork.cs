using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkLibrary
{
    public class ArtificialNeuralNetwork
    {
        private double totalCost;

        public List<Layer> Layers { get; } = new List<Layer>();

        public double TotalCost => totalCost;

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

        public void Train(IDataSource dataSource)
        {
            RecalculateTotalCost(dataSource);
        }

        /// <summary>
        /// Calculates total cost for network from all training data points
        /// </summary>
        /// <param name="dataSource"></param>
        /// <returns>The cost</returns>
        private void RecalculateTotalCost(IDataSource dataSource)
        {
            totalCost = 0.0;

            foreach (var dataPoint in dataSource.DataPoints)
            {
                var estimated = Classify(dataPoint);

                totalCost += CalculateEpsilon(estimated, dataPoint.Label);
            }
        }

        /// <summary>
        /// Epsilon is the error for one data point
        /// </summary>
        /// <param name="dp">The data point</param>
        /// <returns>The error</returns>
        private double CalculateEpsilon(double[] estimated, double[] label)
        {
            // Using sum of squared errors

            var error = 0.0;

            for (int i = 0; i < estimated.Length; i++)
            {
                error += 0.5 * Math.Pow(label[i] - estimated[i], 2);
            }

            return error;
        }

        /// <summary>
        /// Forward-propagate a data point through the ANN.
        /// </summary>
        /// <param name="input">DataPoint with input values, not including bias. The number of dimensions must match the configured ANN.</param>
        /// <returns>The output from the ANN.</returns>
        public double[] Classify(DataPoint input)
        {
            foreach (var neuron in Layers[0].Neurons)
            {
                for (int i = 0; i < input.Value.Length; i++)
                {
                    // Put the value of each input value to the incoming connections of all the layer's neurons
                    // Note: The first incoming connection to a neuron is the bias/threshold, which is not changed.
                    neuron.IncomingConnections[i + 1].Activation = input.Value[i];
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
