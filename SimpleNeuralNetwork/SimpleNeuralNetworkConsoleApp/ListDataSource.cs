using SimpleNeuralNetworkLibrary;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkConsoleApp
{
    public class ListDataSource : IDataSource
    {
        public int InputDimensions { get; set; }
        public int OutputDimensions { get; set; }

        public List<DataPoint> DataPointList { get; } = new List<DataPoint>();
        public IEnumerable<DataPoint> DataPoints => DataPointList.AsEnumerable();
    }
}
