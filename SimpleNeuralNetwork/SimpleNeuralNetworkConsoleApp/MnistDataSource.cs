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
        public string TrainingDataGzPath { get; }
        public string TrainingLabelsGzPath { get; }
        public string TestDataGzPath { get; }
        public string TestLabelsGzPath { get; }
        public IEnumerable<TestCase> TrainingData { get; }
        public IEnumerable<TestCase> TestData { get; }

        public MnistDataSource(string trainingDataGzPath, string trainingLabelsGzPath, string testDataGzPath, string testLabelsGzPath)
        {
            TrainingDataGzPath = trainingDataGzPath;
            TrainingLabelsGzPath = trainingLabelsGzPath;
            TestDataGzPath = testDataGzPath;
            TestLabelsGzPath = testLabelsGzPath;

            TrainingData = FileReaderMNIST.LoadImagesAndLables(trainingLabelsGzPath, trainingDataGzPath);
            TestData = FileReaderMNIST.LoadImagesAndLables(testLabelsGzPath, testDataGzPath);
        }
    }
}
