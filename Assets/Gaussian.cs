
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public static class GaussianBlur
{
    private static float[] kernel;
    private static float currentStrength;
    private static int currentSize;

    private static float[] GetKernel(int size, float strength)
    {
        if (kernel != null && currentSize == size && currentStrength == strength) return kernel;
        MakeKernel(size, strength);
        return kernel;
    }

    public static void MakeKernel(int size, float strength)
    {
        kernel = new float[size * size];

        int radius = size / 2;
        float sigma = strength;
        float sum = 0.0f;
        float calculatedEuler = 1.0f / (2.0f * math.PI * math.pow(sigma, 2));
        // calculate kernel value
        for (int x = -radius; x <= radius; x++)
        {
            for (int y = -radius; y <= radius; y++)
            {
                float value = calculatedEuler * math.exp(-((x * x) + (y * y)) / (2 * (sigma * sigma)));
                sum += value;
                kernel[(y + radius) * size + (x + radius)] = value;
            }
        }
        // normalize values
        for (int i = 0; i < kernel.Length; i++)
        {
            kernel[i] = kernel[i] / sum;
        }

        currentStrength = strength;
        currentSize = size;
    }

    public static NativeArray<float> Blur(NativeArray<float> input, int heightmapWidth, int kernelSize, float kernelStrength)
    {
        //kernel size needs to be an odd number
        if (kernelSize % 2 == 0) kernelSize++;

        NativeArray<float> output = new NativeArray<float>(input.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        NativeArray<float> blurKernel = new NativeArray<float>(kernelSize * kernelSize, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        blurKernel.CopyFrom(GetKernel(kernelSize, kernelStrength));

        var blurJob = new BlurHeightmapFloatToFloat
        {
            input = input,
            output = output,
            heightmapWidth = heightmapWidth,
            kernel = blurKernel,
            kernelDiameter = kernelSize,
            kernelRadius = kernelSize / 2
        }.Schedule(heightmapWidth * heightmapWidth, 32);
        blurJob.Complete();

        blurKernel.Dispose();
        input.Dispose();

        return output;
    }

    [BurstCompile]
    private struct BlurHeightmapFloatToFloat : IJobParallelFor
    {
        [NativeDisableParallelForRestriction, ReadOnly] public NativeArray<float> input;
        [NativeDisableParallelForRestriction, WriteOnly] public NativeArray<float> output;
        [NativeDisableParallelForRestriction, ReadOnly] public NativeArray<float> kernel;
        [ReadOnly] public int heightmapWidth;
        [ReadOnly] public int kernelRadius;
        [ReadOnly] public int kernelDiameter;

        public void Execute(int index)
        {
            int x = index % heightmapWidth; 
            int z = index / heightmapWidth; 

            output[x + heightmapWidth * z] = GetHeightGaussianBlur(x, z);
        }

        public float GetHeightGaussianBlur(int x, int z)
        {
            float value = 0;
            for (int j = -kernelRadius; j <= kernelRadius; j++)
            {
                for (int i = -kernelRadius; i <= kernelRadius; i++)
                {
                    int sx = x + i;
                    int sy = z + j;

                    if (sx < 0 || sx >= heightmapWidth) sx = x + -i;
                    if (sy < 0 || sy >= heightmapWidth) sy = z + -j;

                    float weight = kernel[(j + kernelRadius) * kernelDiameter + (i + kernelRadius)];
                    value += input[sx + heightmapWidth * sy] * weight;
                }
            }
            return value;
        }
    }
}