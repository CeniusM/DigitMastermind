namespace Guess_the_number;

/*
Todo:
Få lavet en metode til at scrable et canavs.
Få lavet settings til denne scrabler
Få lavet en visualizer, evt have den originale som blå, og den nye som rød
*/

// Just takes in the data and scrables it
class CanvasUtil
{
    public const int Res = 28;

    public static string EncodeCanvas(double[,] canvas)
    {
        string str = "";
        for (int i = 0; i < Res; i++)
        {
            for (int j = 0; j < Res; j++)
                str += canvas[j, i].ToString("0.####") + "|";
            str = str.Remove(str.Length - 1);
            str += ";";
        }
        return str.Replace(",", ".").Remove(str.Length - 1);
    }

    public static double[,] ToCanvas(string data)
    {
        double[,] canvas = new double[Res, Res]; // Initialize the canvas
        string[] lines = data.Split(';');
        for (int i = 0; i < Res; i++)
        {
            string[] nums = lines[i].Split("|");
            for (int j = 0; j < Res; j++)
            {
                canvas[j, i] = double.Parse(nums[j]); // Reverse the replacement for parsing
            }
        }
        return canvas;
    }

    public static (int answer, double[,] canvas, double[,] original)[] MakeScrables(string file)
    {
        string[] strings = File.ReadAllLines(file);

        throw new Exception();
    }
}
