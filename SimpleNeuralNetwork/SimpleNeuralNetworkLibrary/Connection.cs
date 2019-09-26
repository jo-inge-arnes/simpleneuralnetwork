﻿using System;

namespace SimpleNeuralNetworkLibrary
{
    public class Connection
    {
        private static readonly Random _random = new Random();

        private bool _isBias;
        private double _activation;

        public bool IsBias
        {
            get { return _isBias; }

            set
            {
                _isBias = value;

                if (_isBias)
                {
                    _activation = 1.0;
                }
            }
        }

        public double Weight { get; set; } = _random.NextDouble();

        public double Activation
        {
            get
            {
                return _activation;
            }

            set
            {
                if (!_isBias)
                {
                    _activation = value;
                }
            }
        }

        public double Value => Weight * Activation;
        public Neuron Sender { get; set; }
        public Neuron Receiver { get; set; }
    }
}
