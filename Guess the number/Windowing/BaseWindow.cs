using SDL_Sharp;
using System.Diagnostics;

namespace Windowing;

public abstract class BaseWindow : IDisposable
{
    public event EventHandler<WindowInitEventArgs>? InitEvent;
    public event EventHandler<WindowClosingEventArgs>? ClosingEvent;
    public event EventHandler<WindowResizeEventArgs>? ResizeEvent;
    public event EventHandler<WindowKeyPressEventArgs>? KeyPressEvent;
    public event EventHandler<WindowShownEventArgs>? ShownEvent;
    public event EventHandler<WindowHiddenEventArgs>? HiddenEvent;
    public event EventHandler<WindowMouseMotionEventArgs>? MouseMotionEvent;
    public event EventHandler<WindowMouseButtonDownEventArgs>? MouseButtomDownEvent;
    public event EventHandler<WindowMouseButtonUpEventArgs>? MouseButtomUpEvent;
    public event EventHandler<WindowMouseScrollEventArgs>? MouseScrollEvent;
    //public event EventHandler<WindowMouseLeftEventArgs>? MouseLeftEvent;
    public event EventHandler<WindowTextInputEventArgs>? TextInputEvent;
    public event EventHandler<WindowCycleDoneEventArgs>? CycleDoneEvent;

    public bool Running { get; private set; }
    public bool Disposed { get; private set; }

    public string Title { get; private set; }
    public int Width { get; private set; }
    public int Height { get; private set; }

    public bool Shown { get; private set; }

    /// <summary>
    /// Indicates the cursors x value: -1 if outside of window
    /// </summary>
    public int MouseX { get; private set; }

    /// <summary>
    /// Indicates the cursors y value: -1 if outside of window
    /// </summary>
    public int MouseY { get; private set; }

    protected IntPtr _window;
    protected IntPtr _renderer;
    protected int _windowID { get; private set; }

    private double _cycleWaitTime;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="width">Width of the window at run time</param>
    /// <param name="height">Height of the window at run time</param>
    public BaseWindow(string title, int width, int height)
    {
        Running = false;
        Disposed = false;

        Title = title;
        Width = width;
        Height = height;

        _window = IntPtr.Zero;
        _renderer = IntPtr.Zero;
        _windowID = -1;

        _cycleWaitTime = 0;
    }

    /// <summary>
    /// Starts the window. This needs to be running on a seperate thread since it is blocking
    /// </summary>
    /// <param name="pullingRate">The pulling rate indicates how many times per second the window checks for events, ]0,100]</param>
    public virtual void Run(int pullingRate)
    {
        if (Disposed)
            throw new Exception("Window have allready been disposed");
        if (pullingRate < 1 || pullingRate > 1000)
            throw new ArgumentException($"{nameof(pullingRate)} should be from 1 and too 1000, but got {pullingRate}");

        lock (this)
        {
            if (Running)
                throw new Exception("Window is allready running");
            Running = true;
        }

        SDL.Init(SdlInitFlags.Everything);

        _cycleWaitTime = (double)1000 / pullingRate;

        _window = SDL.CreateWindow(Title, SDL.WINDOWPOS_CENTERED, SDL.WINDOWPOS_CENTERED, Width, Height, WindowFlags.Resizable);
        _renderer = SDL.CreateRenderer(_window, 0, RendererFlags.Accelerated);
        _windowID = SDL.GetWindowID(_window);

        InitEvent?.Invoke(this, new());

        _Run();
    }

    public void Resize(int w, int h, bool callEvent = false)
    {
        SDL.SetWindowSize(_window, w, h);

        if (callEvent)
            ResizeEvent?.Invoke(this, new(Width, Height, w, h));

        Width = w;
        Height = h;
    }

    private void _Run()
    {
        Stopwatch sw = new Stopwatch();
        sw.Start();

        while (Running)
        {
            sw.Restart();

            while (SDL.PollEvent(out var evt) != 0 && Running)
            {
                if (evt.Type == EventType.Quit)
                {
                    Dispose();
                    return;
                }

                if (evt.Window.Evt == WindowEventID.Leave)
                {
                    MouseX = -1;
                    MouseY = -1;
                }

                if (evt.Type == EventType.KeyDown)
                {
                    KeyPressEvent?.Invoke(this, new((Key)evt.Keyboard.Keysym.Sym));
                }

                if (evt.Type == EventType.MouseWheel)
                {
                    MouseScrollEvent?.Invoke(this, new(evt.Wheel.Y, MouseX, MouseY));
                }

                if (evt.Type == EventType.TextInput)
                {
                    unsafe
                    {

                        string str = System.Text.Encoding.Default.GetString(evt.Text.Text, 32);

                        TextInputEvent?.Invoke(this, new(str[0]));
                    }

                }

                if (evt.Type == EventType.WindowEvent && evt.Window.Evt == WindowEventID.Resized)
                {
                    int oldWidth = Width;
                    int oldHeight = Height;
                    SDL.GetWindowSize(_window, out var w, out var h);
                    int newWidth = w;
                    int newHeight = h;
                    Width = newWidth;
                    Height = newHeight;
                    ResizeEvent?.Invoke(this, new(oldWidth, oldHeight, newWidth, newHeight));
                }

                if (evt.Type == EventType.WindowEvent && evt.Window.Evt == WindowEventID.Shown)
                {
                    Shown = true;
                    ShownEvent?.Invoke(this, new());
                }

                if (evt.Type == EventType.WindowEvent && evt.Window.Evt == WindowEventID.Hidden)
                {
                    Shown = false;
                    HiddenEvent?.Invoke(this, new());
                }

                if (evt.Type == EventType.MouseMotion)
                {
                    MouseX = evt.Motion.X;
                    MouseY = evt.Motion.Y;
                    MouseMotionEvent?.Invoke(this, new(MouseX, MouseY));
                }

                if (evt.Type == EventType.MouseButtonDown)
                {
                    MouseButtomDownEvent?.Invoke(this, new((MouseButton)evt.Button.Button, MouseX, MouseY));
                }

                if (evt.Type == EventType.MouseButtonUp)
                {
                    MouseButtomUpEvent?.Invoke(this, new((MouseButton)evt.Button.Button, MouseX, MouseY));
                }
            }

            double timeToWait = _cycleWaitTime - sw.Elapsed.TotalMilliseconds;

            if (timeToWait > 1 && Running)
                Thread.Sleep((int)timeToWait);

            while (sw.Elapsed.TotalMilliseconds < _cycleWaitTime && Running)
                Thread.Yield();

            if (Running)
                CycleDoneEvent?.Invoke(this, new(sw.Elapsed.TotalSeconds));
        }
    }

    public virtual void Dispose()
    {
        if (Disposed)
            return;
        Disposed = true;

        ClosingEvent?.Invoke(this, new());
        Running = false;

        SDL.DestroyWindow(_window);
        SDL.DestroyRenderer(_renderer);

        SDL.Quit();
    }
}
