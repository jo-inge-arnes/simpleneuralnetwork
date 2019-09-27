using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkLibrary
{
    public class Layer
    {
        public List<Neuron> Neurons { get; } = new List<Neuron>();

        public Layer(int numNeurons, Layer previousLayer = null)
        {
            for (int i = 0; i < numNeurons; i++)
                Neurons.Add(new Neuron());

            if (previousLayer != null)
            {
                FullyConnect(previousLayer);
            }
        }

        public void FullyConnect(Layer previousLayer)
        {
            previousLayer.Neurons.ForEach(previousNeuron => Neurons.ForEach(neuron =>
            {
                var connection = new Connection
                {
                    Receiver = neuron,
                    Sender = previousNeuron
                };

                neuron.IncomingConnections.Add(connection);
                previousNeuron.OutgoingConnections.Add(connection);
            }));
        }

        public void CreateInputConnections(int numInputs)
        {
            Neurons.ForEach(neuron =>
            {
                for (int i = 0; i < numInputs; i++)
                    neuron.IncomingConnections.Add(new Connection { Receiver = neuron });
            });
        }

        public void CreateOutputConnections()
        {
            Neurons.ForEach(neuron => neuron.OutgoingConnections.Add(new Connection { Sender = neuron }));
        }

        /// <summary>
        /// Gets all activation values from all neurons in the layer.
        /// Returns the values as an array of doubles.
        /// </summary>
        /// <returns>An array of the values</returns>
        public double[] AllOutputs()
        {
            return Neurons.SelectMany(n => n.OutgoingConnections).Select(c => c.Activation).ToArray();
        }
    }
}
