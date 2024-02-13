using Guess_the_number;
using Windowing;

const int MaxWindSize = 800;

const int Res = CanvasUtil.Res; // Pixels on each axis
//const int Size = 50; // The size of each pixel
const int PixelSize = MaxWindSize / Res;
const int WindSize = PixelSize * Res;
const int CursorSize = 10;

DrawableWindow window = new DrawableWindow("Draw a number", WindSize, WindSize);
Guessur guessur = new Guessur();
Console.CursorVisible = false;

double PaintStrength = 2;
double InvPaintSize = 3; // Radius
bool Painting = false;
bool Removing = false;
double[,] Canvas = new double[Res, Res];
bool MakingAGuess = false;

// Run async
void MakeGuess(string data)
{
    if (MakingAGuess)
        return;
    MakingAGuess = true;

    if (guessur.Guessing)
        return;

    string guess = guessur.Guess(data);

    Console.SetCursorPosition(0, 0);
    Console.WriteLine(guess);

    MakingAGuess = false;
}

void PaintEvent(int mouseX, int mouseY)
{
    // Draw
    for (int x = 0; x < Res; x++)
    {
        for (int y = 0; y < Res; y++)
        {
            // Calculate distance between the current pixel and the mouse cursor
            double distance = Math.Sqrt(Math.Pow(mouseX / (double)PixelSize - x, 2) + Math.Pow(mouseY / (double)PixelSize - y, 2));

            // Calculate paint strength based on distance
            double strength = Math.Clamp(PaintStrength * Math.Pow(Math.E, -distance * InvPaintSize), 0, 1);
            if (strength > 0.001)
            {
                if (Removing)
                    // Remove paint
                    Canvas[x, y] -= strength;
                else
                    // Add paint
                    Canvas[x, y] += strength;
            }

            // Clamp
            Canvas[x, y] = Math.Clamp(Canvas[x, y], 0, 1);
        }
    }

    // Make new guess
    string data = CanvasUtil.EncodeCanvas(Canvas);
    Task.Run(() => MakeGuess(data));
}

void DrawCanvas()
{
    for (int i = 0; i < Res; i++)
    {
        for (int j = 0; j < Res; j++)
        {
            byte strength = (byte)(Canvas[i, j] * 255);
            window.DrawSquare(
                i * PixelSize, j * PixelSize, // Coord
                PixelSize, PixelSize, // Size
                strength, strength, strength); // Color
        }
    }
}

void Draw()
{
    window.ClearScreen(0, 0, 0);
    DrawCanvas();

    // Draw mouse cursor
    if (window.MouseX != -1)
    {
        // We offset by half of the cursor size
        // and move with half pixel size so the cursor lines up with the drawing
        int size = (int)(CursorSize * PaintStrength);
        int cx = window.MouseX - size / 2 + PixelSize / 2;
        int cy = window.MouseY - size / 2 + PixelSize / 2;
        window.DrawSquare(cx, cy, size, size, 250, 100, 100);
    }

    window.RenderToScreen();
}

void ShowTestData()
{
    CanvasRandomizer randomizer = new CanvasRandomizer(
        maxRotation: 30,
        maxMovement: 5,
        maxZoom: 0.25,
        noiseStrength: 0.3
        );

    var data = CanvasUtil.MakeScrables("C:\\temp\\mnistdata.txt", randomizer);

    foreach (var item in data)
    {
        var answer = item.answer;
        var c1 = item.original;
        var c2 = item.canvas;

        Console.WriteLine("This is " + answer);

        for (int i = 0; i < Res; i++)
        {
            for (int j = 0; j < Res; j++)
            {
                byte b = (byte)(c1[i, j] * 255);
                byte r = (byte)(c2[i, j] * 255);

                window.DrawSquare(
                    i * PixelSize, j * PixelSize, // Coord
                    PixelSize, PixelSize, // Size
                    r, 0, b); // Color
            }
        }

        window.RenderToScreen();

        Console.ReadLine();
    }
}

window.CycleDoneEvent += Window_CycleDoneEvent;
window.MouseButtomDownEvent += Window_MouseButtomDownEvent;
window.MouseButtomUpEvent += Window_MouseButtomUpEvent;
window.MouseMotionEvent += Window_MouseMotionEvent;
window.KeyPressEvent += Window_KeyPressEvent;
window.MouseScrollEvent += Window_MouseScrollEvent;

void Window_MouseScrollEvent(object? sender, WindowMouseScrollEventArgs e)
{
    PaintStrength += 0.5 * e.Scroll;
    PaintStrength = Math.Clamp(PaintStrength, 0.5, 5);
}

window.InitEvent += Window_InitEvent;

void Window_InitEvent(object? sender, WindowInitEventArgs e)
{
    SDL_Sharp.SDL.ShowCursor(0);
}

void Window_KeyPressEvent(object? sender, WindowKeyPressEventArgs e)
{
    if (e.Key == Key.C)
        Array.Clear(Canvas);
    if (e.Key == Key.R)
        Console.Clear();
    if (e.Key == Key.P)
        Console.WriteLine(CanvasUtil.EncodeCanvas(Canvas));
    if (e.Key == Key.T)
        ShowTestData();
    if (e.Key == Key.L)
    {
        string str = Console.ReadLine()!;
        Canvas = CanvasUtil.ToCanvas(str);
    }
}

void Window_CycleDoneEvent(object? sender, WindowCycleDoneEventArgs e)
{
    Draw();
}

void Window_MouseButtomDownEvent(object? sender, WindowMouseButtonDownEventArgs e)
{
    Painting = e.MouseButton == MouseButton.Left;
    Removing = e.MouseButton == MouseButton.Right;
}
void Window_MouseButtomUpEvent(object? sender, WindowMouseButtonUpEventArgs e)
{
    Painting = false;
    Removing = false;
}

void Window_MouseMotionEvent(object? sender, WindowMouseMotionEventArgs e)
{
    if (Painting || Removing)
        PaintEvent(e.X, e.Y);
}

window.Run(200);

guessur.KillProcess();