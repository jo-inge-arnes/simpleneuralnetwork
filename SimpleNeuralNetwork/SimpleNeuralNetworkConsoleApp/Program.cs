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
            var dataSource = InitDataSource();

            var numDimensions = 28*28;
            var numClasses = 10;

            var config = new ArtificialNeuralNetworkConfig
            {
                InputDimensions = numDimensions,
                NeuronCounts = new int[] { numDimensions, numDimensions, numClasses },
                Mu = 0.01
            };

            var ann = new ArtificialNeuralNetwork(config);

            ann.Train(dataSource, 500);
        }

        static IDataSource InitDataSource()
        {
            //var dataSource = new ListDataSource();

            //double[] omega1 = { 1.0, 0.0, 0.0 };
            //double[] omega2 = { 0.0, 1.0, 0.0 };
            //double[] omega3 = { 0.0, 0.0, 1.0 };

            //var normal1 = new Normal(1.0, 0.5);
            //var normal2 = new Normal(5.0, 0.5);
            //var normal3 = new Normal(10.0, 0.5);

            //Random rnd = new Random();

            //for (int i = 0; i < 1000; i++)
            //{
            //    var selectedOmega = rnd.Next(0, 3);
            //    if (selectedOmega == 0)
            //        AddDataPoint(dataSource, omega1, new double[] { normal1.Sample(), normal1.Sample() });
            //    else if(selectedOmega == 1) 
            //        AddDataPoint(dataSource, omega2, new double[] { normal2.Sample(), normal2.Sample() });
            //    else
            //        AddDataPoint(dataSource, omega3, new double[] { normal3.Sample(), normal3.Sample() });
            //}

            var dataSource = new MnistDataSource(
                @"E:\data\train-images-idx3-ubyte.gz",
                @"E:\data\train-labels-idx1-ubyte.gz");

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
