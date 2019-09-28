using MNIST.IO;
using SimpleNeuralNetworkLibrary;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkConsoleApp
{
    public class MnistDataSource : IDataSource
    {
        private List<DataPoint> dataPoints;

        public MnistDataSource(string imagesGzPath, string labelsGzPath)
        {
            var mnistData = FileReaderMNIST.LoadImagesAndLables(labelsGzPath, imagesGzPath);

            dataPoints = mnistData.Select(testCase =>
                new DataPoint { Value = ConvertImage(testCase.Image), Label = ConvertLabel(testCase.Label) }).ToList();
        }

        public IEnumerable<DataPoint> DataPoints
        {
            get
            {
                return dataPoints.AsEnumerable();
            }
        }


        private double[] ConvertImage(byte[,] image)
        {
            return image.Cast<byte>().Select(v => (double)v).ToArray();
        }

        private double[] ConvertLabel(byte label)
        {
            double[] expectedOutputs = new double[10];
            expectedOutputs[label] = 1.0;
            return expectedOutputs;
        }
    }
}
