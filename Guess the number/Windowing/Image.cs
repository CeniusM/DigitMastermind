using SDL_Sharp;

namespace Windowing;

public class Image
{
    public static readonly Image Empty = new Image();

    internal Texture TexturePtr;
    internal int Width;
    internal int Height;
    internal Rect Rect;
    public bool IsEmpty => Width == -1;

    internal Image()
    {
        Width = -1;
        Height = -1;
        Rect = new Rect(0, 0, 0, 0);
    }

    internal Image(Texture texture)
    {
        SetImage(texture);
    }

    internal void SetImage(Texture texture)
    {
        TexturePtr = texture;
        SDL.QueryTexture(texture, out var format, out var access, out var w, out var h);

        Width = w;
        Height = h;

        Rect = new Rect(0, 0, w, h);
    }
}
