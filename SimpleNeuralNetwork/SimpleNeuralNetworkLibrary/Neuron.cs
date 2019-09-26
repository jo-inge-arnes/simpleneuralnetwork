using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkLibrary
{
    public class Neuron
    {
        public List<Connection> IncomingConnections { get; } = new List<Connection>();

        public List<Connection> OutgoingConnections { get; } = new List<Connection>();

        public double Potential => IncomingConnections.Select(c => c.Activation * c.Weight).Sum();

        public double Activation => Math.Max(Potential, 0.0); // ReLU

        public double ActivationDerived => Potential < 0.0 ? 0.0 : 1.0; // ReLU derived

        public Neuron()
        {
            // The first incoming connection is always the bias/threshold
            var bias = new Connection
            {
                IsBias = true,
                Receiver = this
            };

            IncomingConnections.Add(bias);
        }
    }
}
