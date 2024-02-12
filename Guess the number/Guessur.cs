using System.Diagnostics;

namespace Guess_the_number
{
    class Guessur
    {
        private Process _proc;
        public bool Guessing { get; private set; }

        public Guessur()
        {
            _proc = new Process();
            _proc.StartInfo.FileName
                = "C:\\Users\\ceniu\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
            _proc.StartInfo.UseShellExecute = false;
            //_proc.StartInfo.CreateNoWindow = true;
            _proc.StartInfo.RedirectStandardInput = true;
            _proc.StartInfo.RedirectStandardOutput = true;
            _proc.StartInfo.Arguments = "C:\\temp\\digit_rec.py";
            if (!_proc.Start())
                throw new Exception();
        }

        public string Guess(string data)
        {
            if (Guessing)
                throw new Exception("Allready guessing");
            Guessing = true;
            // Write data to the process
            _proc.StandardInput.WriteLine(data);
            _proc.StandardInput.Flush();

            // Read response from the process
            string output = "";
            while (true)
            {
                string response = _proc.StandardOutput.ReadLine()!;

                if (response == "done")
                    break;
                output += response + "\n";
            }

            Guessing = false;
            return output;
        }

        public void KillProcess()
        {
            _proc.Kill();
            _proc.Dispose();
        }
    }
}