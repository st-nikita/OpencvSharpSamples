using System;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Face;
using WarpHelpers;

namespace FaceSwap
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var sourceMat = new Mat("source.jpg"))
            using (var destinationMat = new Mat("destination.jpg"))
            using (var hc = new CascadeClassifier("HaarCascade.xml"))
            using (var facemark = FacemarkLBF.Create())
            {
                Console.WriteLine("Face detection starting..");
                var sourceFaceRects = hc.DetectMultiScale(sourceMat);
                if (sourceFaceRects == null)
                {
                    Console.WriteLine($"Source image: No faces detected.");
                    return;
                }
                Console.WriteLine($"Source image: detected {sourceFaceRects.Length} faces.");

                var destFaceRects = hc.DetectMultiScale(destinationMat);
                if (destFaceRects == null)
                {
                    Console.WriteLine($"Destination image: No faces detected.");
                    return;
                }
                Console.WriteLine($"Destination image: detected {destFaceRects.Length} faces.");
                
                facemark.LoadModel("lbfmodel.yaml");
                using (var sourceInput = InputArray.Create(sourceFaceRects))
                using(var destInput = InputArray.Create(destFaceRects))
                {
                    facemark.Fit(sourceMat, sourceInput, out Point2f[][] sourceLandmarks);
                    var sourcePoints = sourceLandmarks[0];

                    facemark.Fit(destinationMat, destInput, out Point2f[][] destLandmarks);
                    var destPoints = destLandmarks[0];

                    var triangles = destPoints.Take(60).GetDelaunayTriangles();
                    var warps = triangles.GetWarps(sourcePoints.Take(60), destPoints.Take(60));

                    using (var warpedMat = sourceMat.ApplyWarps(destinationMat.Width, destinationMat.Height, warps))
                    using (var mask = new Mat(destinationMat.Size(), MatType.CV_8UC3))
                    using(var result = new Mat(destinationMat.Size(), MatType.CV_8UC3))
                    {
                        mask.SetTo(0);

                        var convexHull = Cv2.ConvexHull(destPoints).Select(s => new Point(s.X, s.Y));
                        Cv2.FillConvexPoly(mask, convexHull, Scalar.White);

                        var rect = Cv2.BoundingRect(convexHull);
                        var center = new Point(rect.X + rect.Width / 2, rect.Y + rect.Height / 2);

                        Cv2.SeamlessClone(warpedMat, destinationMat, mask, center, result, SeamlessCloneMethods.NormalClone);
                        var blured = result.MedianBlur(5);
                        blured.SaveImage("result.png");
                    }
                }
            }
            Console.WriteLine("Done");
        }
    }
}
