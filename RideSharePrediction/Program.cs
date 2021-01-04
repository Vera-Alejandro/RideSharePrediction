using Microsoft.ML;
using RideSharePrediction.DataStructures;
using System;
using System.IO;
using System.Linq;

namespace RideSharePrediction
{
    public class Program
    {
        static string BaseDatasetPath = @"../../../Data";
        static string assetPath = GetAbsolutePath(BaseDatasetPath);

        private static string GetAbsolutePath(string baseDatasetPath)
        {
            throw new NotImplementedException();
        }

        static string fullDatasetPath = Path.Combine(assetPath, "input", "cab_rides.csv");
        static string trainDatasetPath = Path.Combine(assetPath, "output", "trainData.csv");
        static string testDatasetPath = Path.Combine(assetPath, "output", "testData.csv");
        static string ModelPath = Path.Combine(assetPath, "output", "model.zip");

        public static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            PrepareData(mlContext, fullDatasetPath, trainDatasetPath, testDatasetPath);

            IDataView trainingData = mlContext.Data.LoadFromTextFile<RideTransaction>(trainDatasetPath, separatorChar: ',', hasHeader: true);
            IDataView testingData = mlContext.Data.LoadFromTextFile<RideTransaction>(testDatasetPath, separatorChar: ',', hasHeader: true);

            string[] features = trainingData.Schema.AsQueryable()
                                                    .Select(column => column.Name)
                                                    .Where(name => name != "CabType")
                                                    .Where(name => name != "Destination")
                                                    .Where(name => name != "Source")
                                                    .Where(name => name != "Id")
                                                    .Where(name => name != "ProductId")
                                                    .Where(name => name != "Name")
                                                    .ToArray();

            IDataView filteredTrainingData = mlContext.Data.FilterRowsByMissingValues(trainingData, features);

            ITransformer model = TrainModel(mlContext, filteredTrainingData, testingData);

            TestPredictions(mlContext);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        private static void TestPredictions(MLContext mlContext)
        {
            RideTransaction[] ridePredictions =
               {
                new RideTransaction
                {
                    Distance = 0.56f,
                    CabType = "Uber",
                    TimeStamp = 1543302743503,
                    Destination = "Haymarket Square",
                    Source = "North Station",
                    Price = 0, // This is the feild to predict -- Actual: 7.5
                    SurgeMultiplier = 1,
                    Id = "f3d32df3-0105-4740-a95c-7ddd697d6805",
                    ProductId = "55c66225-fbe7-4fd5-9072-eab1ece5e23e",
                    Name = "UberX"
                },
                new RideTransaction
                {
                    Distance = 3.07f,
                    CabType = "Uber",
                    TimeStamp = 1543458860117,
                    Destination = "North Station",
                    Source = "Fenway",
                    Price = 0, // This is the feild to predict -- Actual: 12
                    SurgeMultiplier = 1,
                    Id = "1f384d44-39fa-4537-b6a9-e97c0420fc48",
                    ProductId = "55c66225-fbe7-4fd5-9072-eab1ece5e23e",
                    Name = "UberX"
                },
                new RideTransaction
                {
                    Distance = 1.08f,
                    CabType = "Lyft",
                    TimeStamp = 1544728207629,
                    Destination = "Northeastern University",
                    Source = "Back Bay",
                    Price = 0, // This is the feild to predict -- Actual: 5
                    SurgeMultiplier = 1,
                    Id = "34b3af78-c4bf-431c-80a6-207d80b98539",
                    ProductId = "lyft_line",
                    Name = "Shared"
                },
                new RideTransaction
                {
                    Distance = 0.49f,
                    CabType = "Uber",
                    TimeStamp = 1543258929513,
                    Destination = "North Station",
                    Source = "Haymarket Square",
                    Price = 0, // This is the feild to precit -- Actual: 9.5
                    SurgeMultiplier = 1,
                    Id = "5dc70d45-4924-4d22-8618-62197f9b148b",
                    ProductId = "9a0e7b09-b92b-4c41-9779-2ad22b4d779d",
                    Name = "WAV"
                },
                new RideTransaction
                {
                    Distance = 2.81f,
                    CabType = "Lyft",
                    TimeStamp = 1543453720651,
                    Destination = "West End",
                    Source = "Fenway",
                    Price = 0, // This is the feild to precit -- Actual: 7
                    SurgeMultiplier = 1,
                    Id = "32da8211-493f-44f7-bd1f-ba7b26060fc0",
                    ProductId = "lyft_line",
                    Name = "Shared"
                }
            };

            double[] actualFares = { 7.5, 12, 5, 9.5, 7 };
            int index = 0;

            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<RideTransaction, RideSharePrediction>(trainedModel);
            foreach (var prediction in ridePredictions)
            {
                var resultprediction = predEngine.Predict(prediction);

                Console.WriteLine($"**********************************************************************");
                Console.WriteLine($"Predicted fare: {resultprediction.RideAmount:0.####}, actual fare: {actualFares[index]}");
                Console.WriteLine($"**********************************************************************");
                index++;
            }
        }

        private static ITransformer TrainModel(MLContext mlContext, IDataView filteredTrainingData, IDataView testingData)
        {
            IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(RideTransaction.Price))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CabTypeEncoded", inputColumnName: nameof(RideTransaction.CabType)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DestinationEncoded", inputColumnName: nameof(RideTransaction.Destination)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SourceEncoded", inputColumnName: nameof(RideTransaction.Source)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ProductIdEncoded", inputColumnName: nameof(RideTransaction.ProductId)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "NameEncoded", inputColumnName: nameof(RideTransaction.Name)))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: nameof(RideTransaction.Distance)))
                .Append(mlContext.Transforms.DropColumns(new string[] { "Id" }))
                .Append(mlContext.Transforms.Concatenate("Features", "CabTypeEncoded", "DestinationEncoded", "SourceEncoded", "ProductIdEncoded", "NameEncoded", nameof(RideTransaction.Distance)));


            var trainer = mlContext.Regression.Trainers.FastTree(numberOfLeaves: 15,
                                                                 numberOfTrees: 50,
                                                                 minimumExampleCountPerLeaf: 10,
                                                                 learningRate: .1);


            IEstimator<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("---Training Model -----");
            var trainedModel = trainingPipeline.Fit(trainingData);

            IDataView predictions = trainedModel.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Common.ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;
        }

        private static void PrepareData(MLContext mlContext, string fullDatasetPath, string trainDatasetPath, string testDatasetPath)
        {
            throw new NotImplementedException();
        }
    }
}
