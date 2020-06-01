using System;
using OpenCvSharp;
using OpenCvSharp.Text;

namespace TextRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            var imagePath = @"image.jpg";
            var tessDataPath = @"";

            var engine = OCRTesseract.Create(tessDataPath, null, null, 3, 3);
            var mat = new Mat(imagePath);

            var gray = mat.CvtColor(ColorConversionCodes.BGR2GRAY);
            var canny = gray.Canny(10, 200);
            canny.SaveImage("canny.png");

            var weighted = new Mat(gray.Size(), MatType.CV_8UC1);
            Cv2.AddWeighted(canny, 0.1, gray, 0.9, 0, weighted);
            weighted.SaveImage("weighted.png");

            engine.Run(weighted, out string text, out Rect[] rects, out string[] cText, out float[] conf,
                ComponentLevels.TextLine);

            var mask1 = new Mat(mat.Size(), MatType.CV_8UC1);
            mask1.SetTo(0);

            var mask2 = new Mat(mat.Size(), MatType.CV_8UC1);
            mask2.SetTo(0);

            for (int i = 0; i < conf.Length; i++)
            {
                if (conf[i] == 95)
                    continue;

                var temp1 = new Mat(mask1, rects[i]);
                var temp2 = new Mat(mask2, rects[i]);

                var roi = new Mat(weighted, rects[i]);
                var binary = new Mat(roi.Size(), MatType.CV_8UC1);
                Cv2.Threshold(roi, binary, 0, 255, ThresholdTypes.Otsu);
                var blured = binary.GaussianBlur(new Size(5, 5), 5);

                blured.CopyTo(temp1);
                temp2.SetTo(255);
            }

            mask1.SaveImage("mask1.png");
            mask2.SaveImage("mask2.png");

            var res1 = new Mat(mat.Size(), MatType.CV_8UC3);
            var res2 = new Mat(mat.Size(), MatType.CV_8UC3);
            Cv2.Inpaint(mat, mask1, res1, 10, InpaintMethod.NS);
            Cv2.Inpaint(mat, mask2, res2, 10, InpaintMethod.NS);

            Console.WriteLine(text);
            res1.SaveImage("res1.png");
            res2.SaveImage("res2.png");
            Console.ReadLine();
        }
    }
}
