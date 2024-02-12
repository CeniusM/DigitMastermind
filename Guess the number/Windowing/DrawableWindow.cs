using SDL_Sharp;
using SDL_Sharp.Ttf;

namespace Windowing;

public class DrawableWindow : BaseWindow
{
    protected List<Image> _images = new List<Image>();
    protected List<TextFormat> _textFormats = new List<TextFormat>();

    public DrawableWindow(string title, int width, int height) : base(title, width, height)
    {
        ClosingEvent += Dispose;
    }

    public override void Run(int pullingRate)
    {
        TTF.Init();
        base.Run(pullingRate);
    }

    public unsafe Image LoadImageRaw(IntPtr data, int width, int height)
    {
        if (!Running)
            throw new Exception("Can not load resources before the widnow is running");
        SDL.ClearError();

        // Create a surface and load RGBA data into it
        Surface* surface = SDL.CreateRGBSurfaceFrom(data, width, height, 32, width * 4, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);

        if (surface == null)
        {
            string err = SDL.GetErrorString();
            if (err != "")
                throw new Exception(err);
        }

        // Convert the surface to a texture
        IntPtr texture = SDL.CreateTextureFromSurface(_renderer, surface);

        if (texture == IntPtr.Zero)
        {
            string err = SDL.GetErrorString();
            if (err != "")
                throw new Exception(err);
        }

        // Free the surface as it's no longer needed
        SDL.FreeSurface(surface);

        Image img = new Image(texture);
        _images.Add(img);
        return img;
    }

    public Image LoadImage(string path)
    {
        if (!Running)
            throw new Exception("Can not load resources before the widnow is running");
        SDL.ClearError();
        var texture = SDL_Sharp.Image.IMG.LoadTexture(_renderer, path);
        string err = SDL.GetErrorString();
        if (err != "")
            throw new Exception(err);
        Image img = new Image(texture);
        _images.Add(img);
        return img;
    }

    public TextFormat LoadTextFormat(string path)
    {
        if (!Running)
            throw new Exception("Can not load resources before the widnow is running");
        SDL.ClearError();
        var font = TTF.OpenFont(path, 100);
        string err = SDL.GetErrorString();
        if (err != "")
            throw new Exception(err);
        TextFormat tFormat = new TextFormat(font, 100);
        _textFormats.Add(tFormat);
        return tFormat;
    }

    public void DrawSquareF(float x, float y, float w, float h, byte r, byte g, byte b) =>
        DrawSquare((int)x, (int)y, (int)w, (int)h, r, g, b);

    public void DrawSquare(int x, int y, int w, int h, byte r, byte g, byte b)
    {
        SDL.SetRenderDrawColor(_renderer, r, g, b, 255);
        Rect rect = new Rect(x, y, w, h);
        SDL.RenderFillRect(_renderer, ref rect);
    }

    public void DrawImage(Image img, int x, int y, int w, int h)
    {
        Texture texture = img.TexturePtr;

        Rect src = img.Rect;
        Rect dst = new Rect(x, y, w, h);
        SDL.RenderCopy(_renderer, texture, ref src, ref dst);
    }

    public void DrawImageRotated(Image img, int x, int y, int w, int h, float r)
    {
        Texture texture = img.TexturePtr;

        Rect src = img.Rect;
        Rect dst = new Rect(x, y, w, h);
        Point center = new Point(w / 2, h / 2);
        SDL.RenderCopyEx(_renderer, texture, ref src, ref dst, (double)r, ref center, RendererFlip.None);
    }

    public void DrawTextF(TextFormat textFormat, string text, float x, float y, float scale, byte r, byte g, byte b) =>
        DrawText(textFormat, text, (int)x, (int)y, scale, r, g, b);

    public void DrawText(TextFormat textFormat, string text, int x, int y, float scale, byte r, byte g, byte b)
    {
        Font font = textFormat.FontPtr;

        Rect dst = new Rect(x, y, 0, 0);

        var size = GetTextSize(font, text);

        dst.Height = (int)(size.h * scale);
        dst.Width = (int)(size.w * scale);

        dst.X -= dst.Width / 2;
        dst.Y -= dst.Height / 2;

        DrawText(font, text, dst, r, g, b);
    }

    public (int w, int h) GetTextSize(TextFormat font, string text, float scale = 1f) =>
        GetTextSize(font.FontPtr, text, scale);
    private (int w, int h) GetTextSize(Font font, string text, float scale = 1f)
    {
        TTF.SizeText(font, text, out var w, out var h);
        w = (int)(w * scale);
        h = (int)(h * scale);
        return (w, h);
    }

    private unsafe void DrawText(Font font, string text, Rect dst, byte r, byte g, byte b)
    {
        var surfaceMessage = TTF.RenderText_Solid(font, text, new Color(r, g, b, 255));
        IntPtr messageTexture = SDL.CreateTextureFromSurface(_renderer, surfaceMessage);

        SDL.RenderCopy(_renderer, messageTexture, null, &dst);

        SDL.DestroyTexture(messageTexture);
        SDL.FreeSurface(surfaceMessage);
    }

    public void RenderToScreen()
    {
        SDL.RenderPresent(_renderer);
    }

    public void ClearScreen(byte r, byte g, byte b)
    {
        SDL.SetRenderDrawColor(_renderer, r, g, b, 255);
        SDL.RenderClear(_renderer);
    }

    public void Dispose(object? sender, WindowClosingEventArgs e)
    {
        if (Disposed)
            return;
        foreach (var item in _images)
            SDL.DestroyTexture(item.TexturePtr);
        foreach (var item in _textFormats)
            TTF.CloseFont(item.FontPtr);
        _textFormats.Clear();
        _images.Clear();
        TTF.Quit();
        base.Dispose();
    }
}
