using SimpleNeuralNetworkLibrary;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Reading data...");

            var dataSource = new MnistDataSource(
                @"E:\data\train-images-idx3-ubyte.gz",
                @"E:\data\train-labels-idx1-ubyte.gz");
            Console.WriteLine("Finished reading data");

            var numDimensions = 28 * 28;
            var numClasses = 10;

            var config = new ArtificialNeuralNetworkConfig
            {
                InputDimensions = numDimensions,
                NeuronCounts = new int[] { 128, 128, numClasses }
            };

            var ann = new ArtificialNeuralNetwork(config);

            Console.WriteLine("Training...");

            ann.Train(dataSource);

            Console.WriteLine("Finished training");
        }
    }
}
