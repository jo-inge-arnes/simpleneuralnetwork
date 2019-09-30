using MathNet.Numerics.Distributions;
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
        private double _averageCost;

        /// <summary>
        /// Step size for gradient descent/backpropagation
        /// </summary>
        public double LearningRate { get; set; }

        public List<Layer> Layers { get; } = new List<Layer>();

        public double AverageCost => _averageCost;

        public ArtificialNeuralNetwork(ArtificialNeuralNetworkConfig config)
        {
            LearningRate = config.LearningRate;

            foreach (int neuronCount in config.NeuronCounts)
            {
                var prevLayer = Layers.Count > 0 ? Layers[Layers.Count - 1] : null;
                Layers.Add(new Layer(neuronCount, prevLayer));
            }

            Layers[0].CreateInputConnections(config.InputDimensions);
            Layers[Layers.Count - 1].CreateOutputConnections();

            InitWeights(config);
        }

        private void InitWeights(ArtificialNeuralNetworkConfig config)
        {
            switch (config.ActivationType)
            {
                case ActivationTypeEnum.ReLU:
                    InitWeightsReLU();
                    break;

                default:
                    InitWeightsDefault();
                    break;
            }
        }

        private void InitWeightsDefault()
        {
            var normal = new Normal();

            foreach (Layer layer in Layers)
                foreach (var neuron in layer.Neurons)
                    foreach (var connection in neuron.IncomingConnections)
                        connection.Weight = normal.Sample();
        }

        private void InitWeightsReLU()
        {
            var normal = new Normal();

            foreach (Layer layer in Layers)
                foreach (var neuron in layer.Neurons)
                    foreach (var connection in neuron.IncomingConnections)
                    {
                        if (!connection.IsBias)
                        {
                            double fanIn = layer.Neurons.Count * neuron.IncomingConnections.Count; // Fully connected
                            connection.Weight = normal.Sample() * Math.Sqrt(2.0 / fanIn);
                        }
                        else
                        {
                            connection.Weight = 1.0E-10;
                        }
                    }
        }

        public void Train(IDataSource dataSource, int epochs)
        {
            // TODO: add support for mini-batches

            for (int i = 0; i < epochs; i++)
            {
                Console.WriteLine("Epoch: {0}", i);
                Console.WriteLine("Backpropagating...");

                Backpropagate(dataSource);

                Console.WriteLine("Recalculating average cost...");

                RecalculateCost(dataSource);

                Console.WriteLine("Average cost is now: {0}", AverageCost);
            }
        }

        private void Backpropagate(IDataSource dataSource)
        {
            foreach (var dataPoint in dataSource.DataPoints)
            {
                // Updates all values in the ANN to the datapoint
                Classify(dataPoint);

                for (int r = Layers.Count - 1; r >= 0; r--)
                {
                    int numNeuronsInLayer = Layers[r].Neurons.Count;

                    for (int j = 0; j < numNeuronsInLayer; j++)
                    {
                        var neuron = Layers[r].Neurons[j];

                        // Compute error signal δ 

                        double errorSignal;

                        if (r == Layers.Count - 1)
                        {
                            // Output layer is treated different from the rest.

                            // The last layer only has one outgoing connection per neuron, 
                            // which is the estimated output.
                            var estimated = neuron.OutgoingConnections[0].Activation;
                            var expected = dataPoint.Label[j];

                            // Note: The -1.0 is because it's a partial derivative of squared error.
                            // (Some notations put this minus outside the delta-sign for error signal, others do not.)
                            errorSignal = -1.0 * DifferenceError(expected, estimated) * neuron.ActivationDerived;
                        }
                        else
                        {
                            // The error is the sum of error signal * weight for all outgoing connections to next layer
                            // Because we are working backwards through the layers, this has already been calculated in
                            // the previous step.

                            double sumWeightedErrors = 0.0;

                            foreach (var outgoingConnection in neuron.OutgoingConnections)
                            {
                                sumWeightedErrors += outgoingConnection.ErrorSignal * outgoingConnection.Weight;
                            }

                            errorSignal = sumWeightedErrors * neuron.ActivationDerived;
                        }

                        // Make error signal available to previous layer (backpropagate)
                        foreach (var incomingConnection in neuron.IncomingConnections)
                        {
                            incomingConnection.ErrorSignal = errorSignal;
                        }
                    }
                }

                UpdateWeightsOnline(); // Online updating of weights (per point)
            }
        }

        private void UpdateWeightsOnline()
        {
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    foreach (var incomingConnection in neuron.IncomingConnections)
                    {
                        incomingConnection.Weight -= LearningRate * incomingConnection.ErrorSignal * incomingConnection.Activation;
                    }
                }
            }
        }

        /// <summary>
        /// Calculates average error cost for network from all training data points
        /// </summary>
        /// <param name="dataSource"></param>
        /// <returns>The cost</returns>
        public void RecalculateCost(IDataSource dataSource)
        {
            _averageCost = 0.0;

            int numDataPoints = 0;

            foreach (var dataPoint in dataSource.DataPoints)
            {
                numDataPoints++;

                var estimated = Classify(dataPoint);

                _averageCost += SumSquaredErrors(dataPoint.Label, estimated);
            }

            _averageCost /= numDataPoints;
        }

        private double DifferenceError(double target, double output)
        {
            return target - output;
        }

        /// <summary>
        /// Epsilon is the error for one data point
        /// </summary>
        /// <param name="dp">The data point</param>
        /// <returns>The error</returns>
        private double SumSquaredErrors(double[] label, double[] estimated)
        {
            // Using sum of squared errors

            var sumSquaredErrors = 0.0;

            for (int i = 0; i < estimated.Length; i++)
            {
                sumSquaredErrors += 0.5 * Math.Pow(DifferenceError(label[i], estimated[i]), 2);
            }

            return sumSquaredErrors;
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
