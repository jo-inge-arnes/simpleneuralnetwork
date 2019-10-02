using MathNet.Numerics.Distributions;
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
            //var dataSource = InitSingleValueDataSource();
            var dataSource = InitMnistDataSource();
            // var dataSource = InitListDataSource();

            var config = new ArtificialNeuralNetworkConfig
            {
                InputDimensions = dataSource.InputDimensions,
                NeuronCounts = new int[] { 32, dataSource.OutputDimensions },
                LearningRate = 0.001,
                ActivationType = ActivationTypes.ReLU
            };

            var ann = new ArtificialNeuralNetwork(config);

            //ManuallyInitWeightsForSingleValueSource(ann);

            //var fileName = Guid.NewGuid().ToString() + "-Testing.ann";
            //ann.Save(fileName);
            //var ann2 = ArtificialNeuralNetwork.Load(fileName);

            //var initialResult = ann.Classify(dataSource.DataPointList[0]);
            //ann.RecalculateCost(dataSource);
            //Console.WriteLine("Initial cost: {0}", ann.AverageCost);

            ann.Train(dataSource, 50);
        }

        static ListDataSource InitSingleValueDataSource()
        {
            var dataSource = new ListDataSource();

            dataSource.InputDimensions = 2;
            dataSource.OutputDimensions = 2;

            AddDataPoint(dataSource, new double[] { 0.01, 0.99 }, new double[] { 0.05, 0.10 });

            return dataSource;
        }

        static void ManuallyInitWeightsForSingleValueSource(ArtificialNeuralNetwork ann)
        {
            ann.Layers[0].Neurons[0].IncomingConnections[1].Weight = 0.15;
            ann.Layers[0].Neurons[0].IncomingConnections[2].Weight = 0.20;
            ann.Layers[0].Neurons[0].Bias = 0.35;

            ann.Layers[0].Neurons[1].IncomingConnections[1].Weight = 0.25;
            ann.Layers[0].Neurons[1].IncomingConnections[2].Weight = 0.30;
            ann.Layers[0].Neurons[1].Bias = 0.35;

            ann.Layers[1].Neurons[0].IncomingConnections[1].Weight = 0.40;
            ann.Layers[1].Neurons[0].IncomingConnections[2].Weight = 0.45;
            ann.Layers[1].Neurons[0].Bias = 0.60;

            ann.Layers[1].Neurons[1].IncomingConnections[1].Weight = 0.50;
            ann.Layers[1].Neurons[1].IncomingConnections[2].Weight = 0.55;
            ann.Layers[1].Neurons[1].Bias = 0.60;
        }

        static ListDataSource InitListDataSource()
        {
            var dataSource = new ListDataSource();

            dataSource.InputDimensions = 2;
            dataSource.OutputDimensions = 2;

            double[] omega1 = { 1.0, 0.0 };
            double[] omega2 = { 0.0, 1.0 };

            var normal = new Normal();

            Random rnd = new Random();

            for (int i = 0; i < 1000; i++)
            {
                var selectedOmega = rnd.Next(0, 2);

                if (selectedOmega == 0)
                    AddDataPoint(dataSource, omega1, new double[] { normal.Sample() + 10, normal.Sample() + 5.0 });
                else if (selectedOmega == 1)
                    AddDataPoint(dataSource, omega2, new double[] { normal.Sample(), normal.Sample() });
            }

            return dataSource;
        }

        static MnistDataSource InitMnistDataSource()
        {
            var folderPath = @"C:\Projects\simpleneuralnetwork\mnist_data\";

            var dataSource = new MnistDataSource(
                folderPath + "train-images-idx3-ubyte.gz",
                folderPath + "train-labels-idx1-ubyte.gz")
            {
                InputDimensions = 28 * 28,
                OutputDimensions = 10
            };

            return dataSource;
        }

        static void AddDataPoint(ListDataSource dataSource, double[] label, double[] value)
        {
            dataSource.DataPointList.Add(new DataPoint
            {
                Label = label,
                Value = value
            });
        }
    }
}
