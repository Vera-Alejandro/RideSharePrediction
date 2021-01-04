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
        
        [LoadColumn(0)]
        public string CabType;
        
        [LoadColumn(0)]
        public long TimeStamp;
        
        [LoadColumn(0)]
        public string Destination;
        
        [LoadColumn(0)]
        public string Source;
        
        [LoadColumn(0)]
        public int Price;
        
        [LoadColumn(0)]
        public int SurgeMultiplier;
        
        [LoadColumn(0)]
        public string Id;
        
        [LoadColumn(0)]
        public string ProductId;

        [LoadColumn(0)]
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