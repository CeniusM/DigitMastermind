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

    public static double[,] ToCanvasExpandZero(string data)
    {
        double[,] canvas = new double[Res, Res]; // Initialize the canvas
        string[] lines = data.Split(';');
        for (int i = 0; i < Res; i++)
        {
            string[] nums = lines[i].Split("|");
            int index = 0;
            for (int j = 0; j < Res;)
            {
                if (nums[index].StartsWith("!")) // Handling special case
                {
                    int zeroCount = int.Parse(nums[index].Substring(1)); // Extract the number after "!"
                    for (int k = 0; k < zeroCount; k++)
                    {
                        canvas[j, i] = 0;
                        j++; // Move to the next position
                    }
                }
                else
                {
                    canvas[j, i] = double.Parse(nums[index].Replace(".", ",")); // Reverse the replacement for parsing
                    j++; // Move to the next position
                }
                index++; // Move to the next string in nums
            }
        }
        return canvas;
    }

    public static (int answer, double[,] canvas, double[,] original)[] MakeScrables(string file, CanvasRandomizer canvasRandomizer, int copiesOfSame = 10, string save = "")
    {
        string[] strings = File.ReadAllLines(file);
        int count = strings.Length;

        double[][,] originals = new double[count][,];
        int[] answers = new int[count];
        for (int i = 0; i < count; i++)
        {
            answers[i] = strings[i].Take(1).First() - '0';
            originals[i] = ToCanvasExpandZero(new(strings[i].TakeLast(strings[i].Length - 2).ToArray()));
        }

        List<(int answer, double[,] canvas, double[,] original)> items = new List<(int answer, double[,] canvas, double[,] original)>();

        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < copiesOfSame; j++)
            {
                double[,] copy = new double[Res, Res];
                Array.Copy(originals[i], copy, copy.LongLength);
                canvasRandomizer.Scramble(copy);
                items.Add((answers[i], copy, originals[i]));
            }
        }

        if (save != "")
        {
            if (!File.Exists(file))
                File.Create(file);

            List<string> strs = new List<string>();
            foreach (var item in items)
                strs.Add(item.answer + "?" + EncodeCanvas(item.canvas));
            File.WriteAllLines(save, strs.ToArray());
        }

        return items.ToArray();
    }
}
