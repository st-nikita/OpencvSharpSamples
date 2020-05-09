using System;
using System.Collections.Generic;
using System.Text;

namespace WarpHelpers
{
    /// <summary>
    /// This structure stores a single warp from one triangle to another.
    /// </summary>
    public struct Warp
    {
        /// <summary>
        /// The source triangle to warp from.
        /// </summary>
        public Triangle Source;

        /// <summary>
        /// The destination triangle to warp to.
        /// </summary>
        public Triangle Destination;

        /// <summary>
        /// Construct a new Warp instance.
        /// </summary>
        /// <param name="source">The source triangle to warp from.</param>
        /// <param name="destination">The destination triangle to warp to.</param>
        public Warp(Triangle source, Triangle destination)
        {
            Source = source;
            Destination = destination;
        }
    }
}
