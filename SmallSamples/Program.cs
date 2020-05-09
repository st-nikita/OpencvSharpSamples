using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Face;
using WarpHelpers;

namespace SmallSamples
{
    class Program
    {
        static void Main(string[] args)
        {
            var totalSize = new Mat(new Size(166*2 + 166*4, 166 * 4), MatType.CV_8UC3);
            var files = Directory.GetFiles("Thumbs");
            
            var row1 = new Mat(new Size(166 , 166 * 3), MatType.CV_8UC3);
            var row2 = new Mat(new Size(166 , 166 * 3), MatType.CV_8UC3);
            

            Cv2.VConcat(new Mat[]{new Mat(files[0]).Resize(new Size(166, 166)), new Mat(files[1]).Resize(new Size(166, 166)), new Mat(files[2]).Resize(new Size(166, 166)), new Mat(files[3]).Resize(new Size(166, 166)) }, row1 );
            Cv2.VConcat(new Mat[] { new Mat(files[4]).Resize(new Size(166, 166)), new Mat(files[5]).Resize(new Size(166, 166)), new Mat(files[6]).Resize(new Size(166, 166)), new Mat(files[7]).Resize(new Size(166, 166)) }, row2);
            //Cv2.HConcat(new Mat[] { new Mat(files[6]).Resize(new Size(166, 166)), new Mat(files[7]).Resize(new Size(166, 166)), new Mat(files[8]).Resize(new Size(166, 166)), }, row3);



            Cv2.HConcat(new Mat[]{row1, new Mat("result.png").Resize(new Size(166 * 4, 166 * 4)), row2}, totalSize);

            totalSize.SaveImage("total.png");

            var imagesDirectoryPath = "Images";

            var images = Directory.GetFiles(imagesDirectoryPath);
            var facemark = FacemarkLBF.Create();
            facemark.LoadModel("lbfmodel.yaml");

            Size size = new Size(500,500);
            MatType type = MatType.CV_8UC3;

            using (var haarCascade = new CascadeClassifier("HaarCascade.xml"))
            {
                foreach (var image in images)
                {
                    using (var mat = new Mat(image))
                    {
                        var facesRects = haarCascade.DetectMultiScale(mat);
                        var r = facesRects[0];
                        var roi = new Mat(mat, new Rect(r.Left - r.Width / 4, r.Top -r.Height/4, r.Width+ r.Width/2, r.Height +r.Height/2));
                        var resized = roi.Resize(size);
                        resized.SaveImage($"Thumbs/{Path.GetFileNameWithoutExtension(image)}.png");
                    }
                }


                var allFaceMarks = new List<Point2f[]>();
                foreach (var image1 in Directory.GetFiles("Thumbs"))
                {
                    using (var mat = new Mat(image1))
                    {
                        var facesRects = haarCascade.DetectMultiScale(mat);
                        using (var facesRectsArray = InputArray.Create(facesRects))
                        {
                            facemark.Fit(mat, facesRectsArray, out Point2f[][] landmarks);
                            allFaceMarks.Add(landmarks[0]);
                        }

                        
                    }
                }

                var averagePoints = new List<Point2f>();

                for (int i = 0; i < 68; i++)
                {
                    float xSum = 0;
                    float ySum = 0;
                    for (int j = 0; j < allFaceMarks.Count; j++)
                    {
                        var point = allFaceMarks[j][i];
                        xSum += point.X;
                        ySum += point.Y;
                    }
                    averagePoints.Add(new Point2f(xSum/allFaceMarks.Count, ySum/allFaceMarks.Count));
                }

                averagePoints.Add(new Point2f(1,1));
                averagePoints.Add(new Point2f(1, size.Height-1));
                averagePoints.Add(new Point2f(size.Width-1, 1));
                averagePoints.Add(new Point2f(size.Width-1, size.Height-1));
                var destinationTriangles = averagePoints.GetDelaunayTriangles();
                var result = new Mat(size, type);
                result.SetTo(0);
                var delta = 1.0 / allFaceMarks.Count;

               

                foreach (var image in Directory.GetFiles("Thumbs"))
                {
                    using (var mat = new Mat(image))
                    {

                        var facesRects = haarCascade.DetectMultiScale(mat);
                        using (var facesRectsArray = InputArray.Create(facesRects))
                        {

                            facemark.Fit(mat, facesRectsArray, out Point2f[][] landmarks);

                            var current = new List<Point2f>();
                            current.AddRange(landmarks[0]);
                            current.Add(new Point2f(1, 1));
                            current.Add(new Point2f(1, size.Height-1));
                            current.Add(new Point2f(size.Width-1, 1));
                            current.Add(new Point2f(size.Width-1, size.Height-1));

                            var warps = destinationTriangles.GetWarps(current, averagePoints);



                            var warpedImg = mat.ApplyWarps(mat.Width, mat.Height, warps);
                            Cv2.ImShow("1", warpedImg);
                            Cv2.WaitKey(3000);

                            Cv2.AddWeighted(result, 1, warpedImg, delta, 0, result);
                            

                        }


                    }
                }

                result.SaveImage("result.png");
            }

            



            Console.WriteLine("Done.");
        }
    }
}
