using System.Security.Claims;

namespace Guess_the_number;

class CanvasRandomizer
{
    public double MaxRotation;
    public double MaxMovement;
    public double MaxZoom;
    public double NoiseStrength;
    private Random random;

    public CanvasRandomizer(double maxRotation, double maxMovement, double maxZoom, double noiseStrength)
    {
        MaxRotation = maxRotation;
        MaxMovement = maxMovement;
        MaxZoom = maxZoom;
        NoiseStrength = noiseStrength;
        random = new Random();
    }

    public void Rotate(double[,] canvas)
    {
        // Calculate the dimensions of the canvas
        int width = canvas.GetLength(0);
        int height = canvas.GetLength(1);

        // Calculate the center of the canvas
        double centerX = width / 2.0;
        double centerY = height / 2.0;

        // Generate a random rotation angle within the specified bounds
        double rotationAngle = random.NextDouble() * (MaxRotation * 2) - MaxRotation;

        // Convert the rotation angle from degrees to radians
        double angleInRadians = rotationAngle * Math.PI / 180.0;

        // Create a temporary canvas to store the rotated result
        double[,] rotatedCanvas = new double[width, height];

        // Apply rotation to the canvas
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                // Translate the coordinates to the center of the canvas
                double translatedX = x - centerX;
                double translatedY = y - centerY;

                // Apply rotation transformation
                double rotatedX = translatedX * Math.Cos(angleInRadians) - translatedY * Math.Sin(angleInRadians);
                double rotatedY = translatedX * Math.Sin(angleInRadians) + translatedY * Math.Cos(angleInRadians);

                // Translate the coordinates back to the original position
                rotatedX += centerX;
                rotatedY += centerY;

                // Apply boundary checks to ensure rotated coordinates are within canvas boundaries
                int newX = (int)Math.Round(rotatedX);
                int newY = (int)Math.Round(rotatedY);

                if (newX >= 0 && newX < width && newY >= 0 && newY < height)
                {
                    // Assign the color value to the rotated canvas
                    rotatedCanvas[newX, newY] = canvas[x, y];
                }
            }
        }

        // Copy the rotated canvas back to the original canvas
        Array.Copy(rotatedCanvas, canvas, width * height);
    }

    public void Move(double[,] canvas)
    {
        // Calculate the dimensions of the canvas
        int width = canvas.GetLength(0);
        int height = canvas.GetLength(1);

        // Generate random movement offsets within the specified bounds
        double offsetX = random.NextDouble() * (MaxMovement * 2) - MaxMovement;
        double offsetY = random.NextDouble() * (MaxMovement * 2) - MaxMovement;

        // Create a temporary canvas to store the moved result
        double[,] movedCanvas = new double[width, height];

        // Apply movement to the canvas
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                // Calculate the new coordinates after applying movement offsets
                int newX = (int)(x + offsetX);
                int newY = (int)(y + offsetY);

                // Check if the new coordinates are within the canvas boundaries
                if (newX >= 0 && newX < width && newY >= 0 && newY < height)
                {
                    // Assign the color value to the moved canvas
                    movedCanvas[newX, newY] = canvas[x, y];
                }
            }
        }

        // Copy the moved canvas back to the original canvas
        Array.Copy(movedCanvas, canvas, width * height);
    }

    public void Zoom(double[,] canvas)
    {
        // Calculate the dimensions of the canvas
        int width = canvas.GetLength(0);
        int height = canvas.GetLength(1);

        // Calculate the center of the canvas
        double centerX = width / 2.0;
        double centerY = height / 2.0;

        // Generate a random zoom factor within the specified bounds
        double zoomFactor = 1.0 + (random.NextDouble() * (MaxZoom * 2) - MaxZoom);

        // Create a temporary canvas to store the zoomed result
        double[,] zoomedCanvas = new double[width, height];

        // Apply zoom to the canvas
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                // Calculate the distance from the center of the canvas
                double distanceX = x - centerX;
                double distanceY = y - centerY;

                // Scale the coordinates based on the zoom factor
                double zoomedX = centerX + (distanceX * zoomFactor);
                double zoomedY = centerY + (distanceY * zoomFactor);

                // Apply boundary checks to ensure zoomed coordinates are within canvas boundaries
                int x1 = (int)Math.Max(0, Math.Min(width - 1, Math.Floor(zoomedX)));
                int x2 = (int)Math.Max(0, Math.Min(width - 1, Math.Ceiling(zoomedX)));
                int y1 = (int)Math.Max(0, Math.Min(height - 1, Math.Floor(zoomedY)));
                int y2 = (int)Math.Max(0, Math.Min(height - 1, Math.Ceiling(zoomedY)));

                // Interpolate the color value from the original canvas using bilinear interpolation
                double color = BilinearInterpolation(canvas, zoomedX, zoomedY, x1, x2, y1, y2);

                // Assign the interpolated color to the zoomed canvas
                zoomedCanvas[x, y] = color;
            }
        }

        // Copy the zoomed canvas back to the original canvas
        Array.Copy(zoomedCanvas, canvas, width * height);
    }

    // Bilinear interpolation function to interpolate color values from the original canvas
    private double BilinearInterpolation(double[,] canvas, double x, double y, int x1, int x2, int y1, int y2)
    {
        int width = canvas.GetLength(0);
        int height = canvas.GetLength(1);

        double dx = x - x1;
        double dy = y - y1;

        double topLeft = canvas[x1, y1];
        double topRight = (x2 < width) ? canvas[x2, y1] : topLeft;
        double bottomLeft = (y2 < height) ? canvas[x1, y2] : topLeft;
        double bottomRight = (x2 < width && y2 < height) ? canvas[x2, y2] : topLeft;

        double topInterpolation = topLeft * (1 - dx) + topRight * dx;
        double bottomInterpolation = bottomLeft * (1 - dx) + bottomRight * dx;

        double interpolatedColor = topInterpolation * (1 - dy) + bottomInterpolation * dy;

        return interpolatedColor;
    }

    public void Noise(double[,] canvas)
    {
        // Generate noise for each pixel in the canvas
        for (int i = 0; i < canvas.GetLength(0); i++)
        {
            for (int j = 0; j < canvas.GetLength(1); j++)
            {
                double noise = (random.NextDouble() - 0.5) * NoiseStrength;
                canvas[i, j] += noise;
            }
        }
    }

    public void Clamp(double[,] canvas)
    {
        for (int i = 0; i < canvas.GetLength(0); i++)
            for (int j = 0; j < canvas.GetLength(1); j++)
                canvas[i, j] = Math.Clamp(canvas[i, j], 0, 1);
    }

    public void Scramble(double[,] canvas)
    {
        Rotate(canvas);
        Clamp(canvas);
        Move(canvas);
        Clamp(canvas);
        Zoom(canvas);
        Clamp(canvas);
        Noise(canvas);
        Clamp(canvas);
    }
}
