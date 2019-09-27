using SimpleNeuralNetworkLibrary;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = new ArtificialNeuralNetworkConfig
            {
                InputDimensions = 9,
                NeuronCounts = new int[] { 5, 7, 2}
            };

            var ann = new ArtificialNeuralNetwork(config);

            var dataPoint = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
            var result = ann.Classify(dataPoint);
        }
    }
}
