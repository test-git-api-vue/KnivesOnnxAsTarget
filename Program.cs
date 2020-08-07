using KnivesOnnxAsTarget.Core;
using System;

namespace KnivesOnnxAsTarget
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var mb = new ModelBuilder();
            mb.BuildAndSave();
        }
    }
}
