// The following NuGet packages are required:
// Microsoft.ML.OnnxRuntime
// OpenCvSharp4
// OpenCvSharp4.Extensions
// OpenCvSharp4.runtime (Please install the runtime that matches your environment.)

using System.Diagnostics;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

string modelPath = @".\models\hed.onnx";
string inputImagePath = @".\tests\test_images\ninja512.png";
string outputImagePath = @".\output_hed.png";

var sessionOptions = new SessionOptions();

using var session = new InferenceSession(modelPath, sessionOptions);

// BGR / uint8
using Mat src = Cv2.ImRead(inputImagePath, ImreadModes.Color);

int targetWidth = 512;
int targetHeight = 512;
using Mat resizedSrc = new Mat();
Cv2.Resize(src, resizedSrc, new Size(targetWidth, targetHeight));

int h = resizedSrc.Rows;
int w = resizedSrc.Cols;
int c = resizedSrc.Channels(); // 3

int[] inputShape = new int[] { 1, h, w, c };

byte[] inputData = new byte[h * w * c];
System.Runtime.InteropServices.Marshal.Copy(resizedSrc.Data, inputData, 0, inputData.Length);

var inputTensor = new DenseTensor<byte>(new Memory<byte>(inputData), inputShape);

var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", inputTensor)
        };

var sw = Stopwatch.StartNew();
using var results = session.Run(inputs);
sw.Stop();
Console.WriteLine($"ONNX Inference Time: {sw.ElapsedMilliseconds} ms");

// [1, 3, H, W] (RGB float32)
var outputValue = results.First(v => v.Name == "output");
var outputTensor = outputValue.AsTensor<float>();

using Mat dst = new Mat(h, w, MatType.CV_8UC3);

// [3, H, W] -> [H, W, 3]
// RGB -> BGR
// float32 -> uint8
float[] outData = outputTensor.ToArray();

unsafe
{
    byte* dstPtr = (byte*)dst.Data.ToPointer();
    int planeSize = h * w;

    for (int row = 0; row < h; row++)
    {
        for (int col = 0; col < w; col++)
        {
            int pixelIndex = row * w + col;
            int dstIndex = (row * w + col) * 3;

            float r = outData[0 * planeSize + pixelIndex];
            float g = outData[1 * planeSize + pixelIndex];
            float b = outData[2 * planeSize + pixelIndex];

            dstPtr[dstIndex + 0] = (byte)Math.Clamp(b * 255.0f, 0, 255); // B
            dstPtr[dstIndex + 1] = (byte)Math.Clamp(g * 255.0f, 0, 255); // G
            dstPtr[dstIndex + 2] = (byte)Math.Clamp(r * 255.0f, 0, 255); // R
        }
    }
}

Cv2.ImWrite(outputImagePath, dst);
Console.WriteLine($"Saved: {outputImagePath}");
