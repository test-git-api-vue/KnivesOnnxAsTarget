using KnivesOnnxAsTarget.Structures;
using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.Text;

namespace KnivesOnnxAsTarget.Core
{
    public class ModelBuilder
    {
        public void BuildAndSave()
        {
             MLContext mlContext = new MLContext();

            var traindData = PrepareData();
            var trainingDataView = mlContext.Data.LoadFromEnumerable<ModelInput>(traindData);

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                                      .Append(mlContext.Transforms.LoadRawImageBytes("ImageSource_featurized", null, "ImageSource"))
                                      .Append(mlContext.Transforms.CopyColumns("Features", "ImageSource_featurized"));

            ImageClassificationTrainer.Options options = new ImageClassificationTrainer.Options()
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                WorkspacePath = "TrainedModelOut",
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                FinalModelPrefix = "knives_mdl"
            };

            var trainer = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
            
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var model = trainingPipeline.Fit(trainingDataView);
            
            mlContext.Model.Save(model, trainingDataView.Schema, "./TrainedModelOut/MLModel.zip");
        }

       
        private List<ModelInput> PrepareData()
        {
            var trainData = new List<ModelInput>();

            AddTrainData(25, trainData, "./TrainData/1_neg.png", "Negative");
            AddTrainData(25, trainData, "./TrainData/2_neg.png", "Negative");
            AddTrainData(25, trainData, "./TrainData/3_neg.png", "Negative");
            AddTrainData(25, trainData, "./TrainData/4_neg.png", "Negative");

            AddTrainData(25, trainData, "./TrainData/1_pos.png", "Positive");
            AddTrainData(25, trainData, "./TrainData/2_pos.png", "Positive");
            AddTrainData(25, trainData, "./TrainData/3_pos.png", "Positive");
            AddTrainData(25, trainData, "./TrainData/4_pos.png", "Positive");

            return trainData;
        }

        private void AddTrainData(int count, List<ModelInput> traindData, string path, string label)
        {
            for(int i=0; i<count; i++)
            {
                traindData.Add(new ModelInput() {ImageSource = path, Label = label });
            }
        }
    }
}
