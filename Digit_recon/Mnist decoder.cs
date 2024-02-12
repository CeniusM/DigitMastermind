using System;
using System.Collections.Generic;
using System.Linq;

public class Program
{
    public static void Main()
    {
        string inputstring = "5?!28;!28;!28;!28;!28;!12|0.0118|0.0706|0.0706|0.0706|0.4941|0.5333|0.6863|0.102|0.651|1.0|0.9686|0.498|!4;!8|0.1176|0.1412|0.3686|0.6039|0.6667|0.9922|0.9922|0.9922|0.9922|0.9922|0.8824|0.6745|0.9922|0.949|0.7647|0.251|!4;!7|0.1922|0.9333|0.9922|0.9922|0.9922|0.9922|0.9922|0.9922|0.9922|0.9922|0.9843|0.3647|0.3216|0.3216|0.2196|0.1529|!5;!7|0.0706|0.8588|0.9922|0.9922|0.9922|0.9922|0.9922|0.7765|0.7137|0.9686|0.9451|!10;!8|0.3137|0.6118|0.4196|0.9922|0.9922|0.8039|0.0431|!1|0.1686|0.6039|!10;!9|0.0549|0.0039|0.6039|0.9922|0.3529|!14;!11|0.5451|0.9922|0.7451|0.0078|!13;!11|0.0431|0.7451|0.9922|0.2745|!13;!12|0.1373|0.9451|0.8824|0.6275|0.4235|0.0039|!10;!13|0.3176|0.9412|0.9922|0.9922|0.4667|0.098|!9;!14|0.1765|0.7294|0.9922|0.9922|0.5882|0.1059|!8;!15|0.0627|0.3647|0.9882|0.9922|0.7333|!8;!17|0.9765|0.9922|0.9765|0.251|!7;!14|0.1804|0.5098|0.7176|0.9922|0.9922|0.8118|0.0078|!7;!12|0.1529|0.5804|0.898|0.9922|0.9922|0.9922|0.9804|0.7137|!8;!10|0.0941|0.4471|0.8667|0.9922|0.9922|0.9922|0.9922|0.7882|0.3059|!9;!8|0.0902|0.2588|0.8353|0.9922|0.9922|0.9922|0.9922|0.7765|0.3176|0.0078|!10;!6|0.0706|0.6706|0.8588|0.9922|0.9922|0.9922|0.9922|0.7647|0.3137|0.0353|!12;!4|0.2157|0.6745|0.8863|0.9922|0.9922|0.9922|0.9922|0.9569|0.5216|0.0431|!14;!4|0.5333|0.9922|0.9922|0.9922|0.8314|0.5294|0.5176|0.0627|!16;!28;!28;!28";
        var imgdata = DataReverter(inputstring);
        Console.WriteLine(imgdata);
    }

    public static List<List<double>> ZeroExpander(List<string> linenums)
    {
        var newlinenums = new List<string>();
        foreach (var num in linenums)
        {
            if (num.Contains('!'))
            {
                newlinenums.AddRange(Enumerable.Repeat("0", int.Parse(num.Substring(1))));
            }
            else
            {
                newlinenums.Add(num);
            }
        }
        return newlinenums.Select(x => double.Parse(x)).ToList();
    }

    public static Tuple<int, List<List<double>>> DataReverter(string data)
    {
        var splitData = data.Split('?');
        var target = int.Parse(splitData[0]);
        var lines = splitData[1].Split(';');
        var newlines = new List<List<double>>();
        foreach (var line in lines)
        {
            var linenums = line.Split('|').ToList();
            linenums = ZeroExpander(linenums);
            newlines.Add(linenums);
        }
        return new Tuple<int, List<List<double>>>(target, newlines);
    }
}