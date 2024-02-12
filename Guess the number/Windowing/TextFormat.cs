using SDL_Sharp.Ttf;

namespace Windowing;

public class TextFormat
{
    public static readonly TextFormat Empty = new TextFormat();

    internal Font FontPtr;
    internal int Size;

    internal TextFormat()
    {
        FontPtr = IntPtr.Zero;
        Size = 0;
    }

    internal TextFormat(Font font, int size)
    {
        FontPtr = font;
        Size = size;
    }
}
