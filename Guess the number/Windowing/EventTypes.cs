namespace Windowing;

public abstract class BaseWindowEventArgs : EventArgs
{
    public DateTime TimeStamp { get; }

    public BaseWindowEventArgs()
    {
        TimeStamp = DateTime.Now;
    }
}

public class WindowResizeEventArgs : BaseWindowEventArgs
{
    public int OldWidth { get; }
    public int OldHeight { get; }
    public int NewWidth { get; }
    public int NewHeight { get; }

    internal WindowResizeEventArgs(int oldWidth, int oldHeight, int newWidth, int newHeight)
    {
        OldWidth = oldWidth;
        OldHeight = oldHeight;
        NewWidth = newWidth;
        NewHeight = newHeight;
    }
}

public class WindowKeyPressEventArgs : BaseWindowEventArgs
{
    public Key Key { get; }

    internal WindowKeyPressEventArgs(Key key)
    {
        Key = key;
    }
}

public class WindowClosingEventArgs : BaseWindowEventArgs
{
    internal WindowClosingEventArgs()
    {
    }
}

public class WindowShownEventArgs : BaseWindowEventArgs
{
    internal WindowShownEventArgs()
    {
    }
}

public class WindowHiddenEventArgs : BaseWindowEventArgs
{
    internal WindowHiddenEventArgs()
    {
    }
}

public class WindowMouseMotionEventArgs : BaseWindowEventArgs
{
    public int X;
    public int Y;

    internal WindowMouseMotionEventArgs(int x, int y)
    {
        X = x;
        Y = y;
    }
}

public class WindowMouseButtonDownEventArgs : BaseWindowEventArgs
{
    public MouseButton MouseButton;
    public int X, Y;

    internal WindowMouseButtonDownEventArgs(MouseButton mouseButton, int x, int y)
    {
        MouseButton = mouseButton;
        X = x;
        Y = y;
    }
}

public class WindowMouseButtonUpEventArgs : BaseWindowEventArgs
{
    public MouseButton MouseButton;
    public int X, Y;

    internal WindowMouseButtonUpEventArgs(MouseButton mouseButton, int x, int y)
    {
        MouseButton = mouseButton;
        X = x;
        Y = y;
    }
}

public class WindowMouseScrollEventArgs : BaseWindowEventArgs
{
    public int Scroll;
    public int X, Y;

    internal WindowMouseScrollEventArgs(int scroll, int x, int y)
    {
        Scroll = scroll;
        X = x;
        Y = y;
    }
}

public class WindowInitEventArgs : BaseWindowEventArgs
{
    internal WindowInitEventArgs()
    {
    }
}

public class WindowTextInputEventArgs : BaseWindowEventArgs
{
    public char Text;

    internal WindowTextInputEventArgs(char text)
    {
        Text = text;
    }
}

public class WindowCycleDoneEventArgs : BaseWindowEventArgs
{
    public double DeltaTime;

    internal WindowCycleDoneEventArgs(double deltaTime)
    {
        DeltaTime = deltaTime;
    }
}