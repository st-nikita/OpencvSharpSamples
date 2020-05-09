using System;
using OpenCvSharp;

namespace WarpHelpers
{
    public struct Triangle
    {
        /// <summary>
        /// The first triangle corner. 
        /// </summary>
        public Point2f P1;

        /// <summary>
        /// The second triangle corner. 
        /// </summary>
        public Point2f P2;

        /// <summary>
        /// The third triangle corner. 
        /// </summary>
        public Point2f P3;

        /// <summary>
        /// Construct a new Triangle instance.
        /// </summary>
        /// <param name="p1">The first triangle corner.</param>
        /// <param name="p2">The second triangle corner.</param>
        /// <param name="p3">The third triangle corner.</param>
        public Triangle(Point2f p1, Point2f p2, Point2f p3)
        {
            P1 = p1;
            P2 = p2;
            P3 = p3;
        }

        /// <summary>
        /// Convert the triangle to an array of Point2f structures.
        /// </summary>
        /// <returns>An array of Point2f structures corresponding to each triangle corner</returns>
        public Point2f[] ToPoint2f()
        {
            return new Point2f[]
            {
                new Point2f(P1.X, P1.Y),
                new Point2f(P2.X, P2.Y),
                new Point2f(P3.X, P3.Y)
            };
        }
    }
}
