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
            // var dataSource = InitListDataSource();

            var dataSource = InitMnistDataSource();

            //var config = new ArtificialNeuralNetworkConfig
            //{
            //    InputDimensions = dataSource.InputDimensions,
            //    NeuronCounts = new int[] { 32, dataSource.OutputDimensions },
            //    LearningRate = 0.001,
            //    ActivationType = ActivationTypes.ReLU
            //};

            //var ann = new ArtificialNeuralNetwork(config);
            var dir = @"C:\Projects\simpleneuralnetwork\";
            var filename = "d58b64a6-fccb-4662-8aa8-a1c4169dfa03-1235-p-0,0674167965245353.ann";
            var fullPath = dir + filename;

            var ann = ArtificialNeuralNetwork.Load(fullPath);
            ann.LearningRate = 0.00001;
            ann.Train(dataSource, 10000);

            //var ann = ArtificialNeuralNetwork.Load(fullPath);
            //CalculatePercentCorrect(ann);

            Console.WriteLine("Press <enter> to quit...");
            Console.ReadLine();
        }

        static void CalculatePercentCorrect(ArtificialNeuralNetwork ann)
        {
            var testDataSource = InitMnistTesttDataSource();
            int numPoints = 0;
            int numCorrect = 0;

            var classCounts = new Dictionary<int, (int numCorrect, int numTotal)>();  

            foreach (var dataPoint in testDataSource.DataPoints)
            {
                numPoints++;
                var estimated = ann.Classify(dataPoint);
                var expected = dataPoint.Label;

                var classEstimated = getClass(estimated);
                var classExpected = getClass(expected);

                var classCountsTouple =
                    classCounts.TryGetValue(classExpected, out var touple) ? touple : (numCorrect: 0, numTotal: 0);
                    
                classCountsTouple.numTotal++;

                if (classEstimated == classExpected)
                {
                    numCorrect++;
                    classCountsTouple.numCorrect++;
                }

                classCounts[classExpected] = classCountsTouple;
            }

            Console.WriteLine("Percent correct: {0}", (100.0 * numCorrect) / (double)numPoints);
            for (int i = 0; i < 10; i++)
            {
                var classCountsTouple =
                    classCounts.TryGetValue(i, out var touple) ? touple : (numCorrect: 0, numTotal: 0);

                var percent = 100.0 * classCountsTouple.numCorrect / (double)classCountsTouple.numTotal;
                
                Console.WriteLine("{0}: {1}", i, percent);
            }
        }

        static int getClass(double[] outputs)
        {
            int indexClosestToOne = -1;
            double closestSquaredDistance = double.MaxValue;

            for (int i = 0; i < outputs.Length; i++)
            {
                double squaredDistance = Math.Pow(outputs[i] - 1.0, 2);
                if (squaredDistance < closestSquaredDistance)
                {
                    indexClosestToOne = i;
                    closestSquaredDistance = squaredDistance;
                }
            }            

            return indexClosestToOne;
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
            var folderPath = @"C:\Datasets\JPG-PNG-to-MNIST-NN-Format\"; // @"C:\Projects\simpleneuralnetwork\mnist_data\";

            var dataSource = new MnistDataSource(
                folderPath + "train-images-idx3-ubyte.gz",
                folderPath + "train-labels-idx1-ubyte.gz")
            {
                InputDimensions = 28 * 28,
                OutputDimensions = 10
            };

            return dataSource;
        }

        static MnistDataSource InitMnistTesttDataSource()
        {
            var folderPath = @"C:\Datasets\JPG-PNG-to-MNIST-NN-Format\"; // @"C:\Projects\simpleneuralnetwork\mnist_data\";

            var dataSource = new MnistDataSource(
                folderPath + "test-images-idx3-ubyte.gz",
                folderPath + "test-labels-idx1-ubyte.gz")
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
