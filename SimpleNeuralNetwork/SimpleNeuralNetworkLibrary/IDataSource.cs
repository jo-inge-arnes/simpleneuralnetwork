using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkLibrary
{
    public interface IDataSource
    {
        int InputDimensions { get; set; }
        int OutputDimensions { get; set; }

        IEnumerable<DataPoint> DataPoints { get; }
    }
}
