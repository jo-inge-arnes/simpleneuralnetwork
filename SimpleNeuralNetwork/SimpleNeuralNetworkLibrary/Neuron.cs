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

        public double Potential => IncomingConnections.Select(c => c.Weight * c.Activation).Sum();

        private const double a = 100.0;

        public double Activation => F(Potential); // Math.Max(Potential, 0.0); // ReLU

        public double ActivationDerived => a * F(Potential) * (1 - F(Potential)); // Potential > 0.0 ? 1.0 : 0.0; // ReLU derived

        private double F(double x)
        {
            return (1.0 / (1.0 + Math.Exp(-a*x)));
        }
        
        public double Bias 
        {
            get { return IncomingConnections[0].Weight; } 
            set { IncomingConnections[0].Weight = value; } 
        }

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
