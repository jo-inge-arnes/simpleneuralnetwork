using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkLibrary
{
    public class ArtificialNeuralNetwork
    {
        private double totalCost;

        /// <summary>
        /// Step size for gradient descent/backpropagation
        /// </summary>
        public double Mu { get; set; }

        public List<Layer> Layers { get; } = new List<Layer>();

        public double TotalCost => totalCost;

        public ArtificialNeuralNetwork(ArtificialNeuralNetworkConfig config)
        {
            Mu = config.Mu;

            foreach (int neuronCount in config.NeuronCounts)
            {
                var prevLayer = Layers.Count > 0 ? Layers[Layers.Count - 1] : null;
                Layers.Add(new Layer(neuronCount, prevLayer));
            }

            Layers[0].CreateInputConnections(config.InputDimensions);
            Layers[Layers.Count - 1].CreateOutputConnections();
        }

        public void Train(IDataSource dataSource, int epochs)
        {
            // TODO: add support for mini-batches

            for (int i = 0; i < epochs; i++)
            {
                Console.WriteLine("Epoch: {0}", i);
                Console.WriteLine("Backpropagating...");

                Backpropagate(dataSource);

                Console.WriteLine("Updating weights...");

                UpdateWeights();

                Console.WriteLine("Recalculating total cost...");

                RecalculateTotalCost(dataSource);

                Console.WriteLine("Total cost is now: {0}", TotalCost);
            }
        }

        private void UpdateWeights()
        {
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    foreach (var incomingConnection in neuron.IncomingConnections)
                    {
                        incomingConnection.Weight -= (Mu * incomingConnection.WeightGradient);
                        incomingConnection.WeightGradient = 0.0;
                    }
                }
            }
        }

        private void Backpropagate(IDataSource dataSource)
        {
            foreach (var dataPoint in dataSource.DataPoints)
            {
                // Updates all values in the ANN to the datapoint
                Classify(dataPoint);

                int L = Layers.Count - 1;

                for (int r = L; r >= 0; r--)
                {
                    int k_r = Layers[r].Neurons.Count;

                    for (int j = 0; j < k_r; j++)
                    {
                        var neuron = Layers[r].Neurons[j];

                        // Compute error signal δ 

                        double errorSignal;

                        if (r == L)
                        {
                            // Output layer is treated different from the rest.

                            // The last layer only has one outgoing connection per neuron, 
                            // which is the estimated output.
                            var y_hat = neuron.OutgoingConnections[0].Activation;
                            var y = dataPoint.Label[j];

                            errorSignal = neuron.ActivationDerived * e(y, y_hat);
                        }
                        else
                        {
                            // The error is the sum of error signal * weight for all outgoing connections to next layer
                            // Because we are working backwards through the layers, this has already been calculated in
                            // the previous step.

                            double e = 0.0;

                            foreach (var outgoingConnection in neuron.OutgoingConnections)
                            {
                                e += outgoingConnection.ErrorSignal * outgoingConnection.Weight;
                            }

                            errorSignal = e * neuron.ActivationDerived;
                        }

                        // Make error signal available to previous layer (backpropagate)
                        foreach (var incomingConnection in neuron.IncomingConnections)
                        {
                            incomingConnection.ErrorSignal = errorSignal;
                            incomingConnection.WeightGradient += errorSignal * incomingConnection.Activation;
                        }
                    }
                }
            }
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

                totalCost += CalculateEpsilon(dataPoint.Label, estimated);
            }
        }

        private double e(double label, double estimated)
        {
            return label - estimated;
        }

        /// <summary>
        /// Epsilon is the error for one data point
        /// </summary>
        /// <param name="dp">The data point</param>
        /// <returns>The error</returns>
        private double CalculateEpsilon(double[] label, double[] estimated)
        {
            // Using sum of squared errors

            var error = 0.0;

            for (int i = 0; i < estimated.Length; i++)
            {
                error += Math.Pow(e(label[i], estimated[i]), 2);
            }

            return 0.5 * error;
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
            var outputs = Layers[Layers.Count - 1].AllOutputs();

            return outputs;
        }
    }
}
