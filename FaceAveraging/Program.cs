using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Face;
using WarpHelpers;

namespace FaceAveraging
{
    class Program
    {
        private const string ImagesPath = "Images";
        private const string TempDirName = "Temp";
        private static CascadeClassifier _cascade = new CascadeClassifier("HaarCascade.xml");
        private static FacemarkLBF _facemark;
        private static Size _outputSize = new Size(500,500);
        private static MatType _matTypeDefault = MatType.CV_8UC3;

        static void Main(string[] args)
        {
            //prepare images for averaging
            PrepareImages();

            //load facemark model
            _facemark = FacemarkLBF.Create();
            _facemark.LoadModel("lbfmodel.yaml");

            //collection for all found facemarks
            var allFaceMarks = new List<List<Point2f>>();

            //facemark search and save
            foreach (var image in Directory.GetFiles(TempDirName))
            {
                using (var mat = new Mat(image))
                {
                    var facesRects = _cascade.DetectMultiScale(mat);
                    using (var facesRectsArray = InputArray.Create(facesRects))
                    {
                        _facemark.Fit(mat, facesRectsArray, out Point2f[][] landmarks);
                        // only one face should be
                        allFaceMarks.Add(landmarks[0].ToList());
                    }
                }
            }

            //add static points 
            foreach (var facemarks in allFaceMarks)
            {
                facemarks.Add(new Point2f(1, 1));
                facemarks.Add(new Point2f(1, _outputSize.Height/2));
                facemarks.Add(new Point2f(1, _outputSize.Height - 1));
                facemarks.Add(new Point2f(_outputSize.Width - 1, 1));
                facemarks.Add(new Point2f(_outputSize.Width/2, _outputSize.Height - 1));
                facemarks.Add(new Point2f(_outputSize.Width - 1, _outputSize.Height/2));
                facemarks.Add(new Point2f(_outputSize.Width - 1, _outputSize.Height - 1));
            }

            //average Facemarks
            var averagePoints = new List<Point2f>();
            for (int i = 0; i < 75; i++)
            {
                float xSum = 0;
                float ySum = 0;
                for (int j = 0; j < allFaceMarks.Count; j++)
                {
                    var point = allFaceMarks[j][i];
                    xSum += point.X;
                    ySum += point.Y;
                }
                averagePoints.Add(new Point2f(xSum / allFaceMarks.Count, ySum / allFaceMarks.Count));
            }

            //calculate delaunay triangles
            var destinationTriangles = averagePoints.GetDelaunayTriangles();

            //create result mat
            var outputMat = new Mat(_outputSize, _matTypeDefault);
            outputMat.SetTo(0);

            // blending coeff
            var delta = 1.0 / allFaceMarks.Count;

            // warping and blending
            var files = Directory.GetFiles(TempDirName);
            for (int i = 0; i < files.Length; i++)
            {
                using (var mat = new Mat(files[i]))
                {
                    var landmarks = allFaceMarks[i];
                    var warps = destinationTriangles.GetWarps(landmarks, averagePoints);
                    var warpedImg = mat.ApplyWarps(mat.Width, mat.Height, warps);
                    Cv2.AddWeighted(outputMat, 1, warpedImg, delta, 0, outputMat);
                }
            }
            
            //save
            outputMat.SaveImage("result.png");
            Console.WriteLine("Done.");

        }

        private static void PrepareImages()
        {
            var images = Directory.GetFiles(ImagesPath);
            if (!Directory.Exists(TempDirName))
            {
                Directory.CreateDirectory(TempDirName);
            }
            else
            {
                var files = Directory.GetFiles(TempDirName);
                foreach (var file in files)
                {
                    File.Delete(file);
                }
            }

            foreach (var image in images)
            {
                try
                {
                    using (var mat = new Mat(image))
                    {
                        var facesRects = _cascade.DetectMultiScale(mat);

                        if (facesRects == null)
                            continue;

                        if (facesRects.Length < 1)
                            continue;

                        var r = facesRects[0];
                        var roi = new Mat(mat,
                            new Rect(r.Left - r.Width / 4, r.Top - r.Height / 4, r.Width + r.Width / 2,
                                r.Height + r.Height / 2));
                        var resized = roi.Resize(_outputSize);
                        resized.SaveImage($"{TempDirName}/{Path.GetFileNameWithoutExtension(image)}.png");
                    }
                }
                catch (Exception e)
                {
                    continue;
                }
            }
        }
    }
}
