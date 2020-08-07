using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace KnivesOnnxAsTarget.Structures
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public String Prediction { get; set; }
        public float[] Score { get; set; }
    }
}
