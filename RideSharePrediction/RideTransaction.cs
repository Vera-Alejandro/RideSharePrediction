using Microsoft.ML.Data;
using System;

namespace RideSharePrediction.DataStructures
{
    public interface IModelEntity
    {
        void PrintToConsole();
    }

    public class RideTransaction : IModelEntity
    {
        [LoadColumn(0)]
        public float Distance;

        [LoadColumn(1)]
        public string CabType;

        [LoadColumn(2)]
        public float TimeStamp;

        [LoadColumn(3)]
        public string Destination;

        [LoadColumn(4)]
        public string Source;

        [LoadColumn(5)]
        public float Price;

        [LoadColumn(6)]
        public float SurgeMultiplier;

        [LoadColumn(7)]
        public string Id;

        [LoadColumn(8)]
        public string ProductId;

        [LoadColumn(9)]
        public string Name;

        public void PrintToConsole()
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Row View for Transaction Data            ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Step:                   {Distance}       ");
            Console.WriteLine($"*       Type:                   {CabType}        ");
            Console.WriteLine($"*       Amount:                 {TimeStamp}      ");
            Console.WriteLine($"*       NameOrigin:             {Destination}    ");
            Console.WriteLine($"*       OldBalanceOrg:          {Source}         ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"*       NewBalanceOrig:         {Price}          ");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"*       NameDest:               {SurgeMultiplier}");
            Console.WriteLine($"*       OldBalanceDest:         {Id}             ");
            Console.WriteLine($"*       NewBalanceDest:         {ProductId}      ");
            Console.WriteLine($"*       IsFraud:                {Name}           ");
            Console.WriteLine($"*************************************************");
        }
    }
}
