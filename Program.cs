using System;
using Accord.IO;
using Accord.MachineLearning;
using Accord.MachineLearning.Bayes;

namespace dotnetcore
{
    class Program
    {
        static void Main(string[] args)
        {
				int[][] inputs =
				{
					 //               input      output
					 new int[] { 0, 1, 1, 0 }, //  0 
					 new int[] { 0, 1, 0, 0 }, //  0
					 new int[] { 0, 0, 1, 0 }, //  0
					 new int[] { 0, 1, 1, 0 }, //  0
					 new int[] { 0, 1, 0, 0 }, //  0
					 new int[] { 1, 0, 0, 0 }, //  1
					 new int[] { 1, 0, 0, 0 }, //  1
					 new int[] { 1, 0, 0, 1 }, //  1
					 new int[] { 0, 0, 0, 1 }, //  1
					 new int[] { 0, 0, 0, 1 }, //  1
					 new int[] { 1, 1, 1, 1 }, //  2
					 new int[] { 1, 0, 1, 1 }, //  2
					 new int[] { 1, 1, 0, 1 }, //  2
					 new int[] { 0, 1, 1, 1 }, //  2
					 new int[] { 1, 1, 1, 1 }, //  2
				};

				int[] outputs = 
				{
					 0, 0, 0, 0, 0,
					 1, 1, 1, 1, 1,
					 2, 2, 2, 2, 2,
				};

				// Create a learning algorithm
				var learner = new NaiveBayesLearning();

				// Teach a model on the data examples
				NaiveBayes nb = learner.Learn(inputs, outputs);

				// Save and Load model
				nb.Save("./naive-bayes.model");
				NaiveBayes estimator = NaiveBayes.Load("./naive-bayes.model");

				// Output the model 
				Console.WriteLine(estimator.Decide(new int[] { 0, 1, 1, 0 })); 
				Console.WriteLine(estimator.Decide(new int[] { 1, 1, 1, 1 })); 
				Console.WriteLine(estimator.Decide(new int[] { 0, 1, 1, 1 })); 
				Console.WriteLine(estimator.Decide(new int[] { 1, 1, 1, 0 })); 
				Console.WriteLine(estimator.Decide(new int[] { 0, 0, 1, 0 })); 
				Console.WriteLine(estimator.Decide(new int[] { 1, 0, 1, 0 })); 
				Console.WriteLine(estimator.Decide(new int[] { 1, 1, 0, 1 })); 
        }
    }
}

