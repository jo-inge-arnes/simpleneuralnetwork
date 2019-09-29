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

            var numDimensions = 28 * 28;
            var numClasses = 10;

            var config = new ArtificialNeuralNetworkConfig
            {
                InputDimensions = numDimensions,
                NeuronCounts = new int[] { 32, 32, 32, 32, numClasses },
                Mu = 0.01
            };

            var ann = new ArtificialNeuralNetwork(config);

            ann.Train(dataSource, 100);
        }

        static IDataSource InitDataSource()
        {
            //var dataSource = new ListDataSource();

            //AddDataPoint(dataSource, new double[] { 1.0, 0.0, 0.0 }, new double[] { 1.0, 2.0 });
            //AddDataPoint(dataSource, new double[] { 0.0, 1.0, 0.0 }, new double[] { 2.0, 3.0 });
            //AddDataPoint(dataSource, new double[] { 0.0, 0.0, 1.0 }, new double[] { 3.0, 4.0 });


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
