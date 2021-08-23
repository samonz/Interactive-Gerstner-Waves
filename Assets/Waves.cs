using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public struct GerstnerWaveOptions
{
    public GerstnerWaveOptions(float2 dir, float steep, float length)
    {
        direction = dir;
        steepness = steep;
        wavelength = length;
    }
    public float2 direction;   // Direction of the wave
    public float steepness;  // Steepness/Sharpness of the peaks
    public float wavelength; // Wavelength...self explnitory
}
public class Waves : MonoBehaviour
{
    public int oceanWidth = 50;
    public float skirtHeight = 10;
    public int interactionGridWidth = 20;
    public float heightMultiplier = 2f;
    public float damping = 0.98f;
    public float scale = 1;
    public int gridUpdateFrequency = 5;  
    public float speed = 1;
    public float noiseMultiplier = 0.05f;
    public float noiseScale = 0.05f;
    public GameObject collisionPlane;
    NativeArray<float3> vertices;
    NativeArray<float3> originalPositions;
    NativeArray<float> interactionGridA;
    NativeArray<float> interactionGridB;
    NativeArray<int> indices;
    Mesh mesh;
    MeshFilter filter;
    GerstnerWaveOptions WaveA;
    GerstnerWaveOptions WaveB;
    GerstnerWaveOptions WaveC;
    GerstnerWaveOptions WaveD;
    float meshScale;
    long frameCount = 0;
    float t;
    int iCount;
    int vCount;

    void OnEnable()
    {
        //create our buffers
        vertices = new NativeArray<float3>(oceanWidth * oceanWidth, Allocator.Persistent);
        originalPositions = new NativeArray<float3>(oceanWidth * oceanWidth, Allocator.Persistent);
        indices = new NativeArray<int>((oceanWidth - 1) * (oceanWidth - 1) * 6, Allocator.Persistent);
        interactionGridA = new NativeArray<float>(interactionGridWidth * interactionGridWidth, Allocator.Persistent);
        interactionGridB = new NativeArray<float>(interactionGridWidth * interactionGridWidth, Allocator.Persistent);
        //create our gerstner waves
        WaveA = new GerstnerWaveOptions(new float2(0.0f, -1.0f), 0.5f, 2.4f);
        WaveB = new GerstnerWaveOptions(new float2(0.0f, 1.0f), 0.25f, 4.3f);
        WaveC = new GerstnerWaveOptions(new float2(1.0f, 1.0f), 0.15f, 6.2f);
        WaveD = new GerstnerWaveOptions(new float2(1.0f, 1.0f), 0.4f, 2.1f);
        //scale the mesh to fit the screen
        meshScale = 1.0f / (oceanWidth * 0.1f);
        //create the mesh original vertices and indices
        CreateMesh();
        scale = 1;
    }

    private void OnDisable()
    {
        if (vertices.IsCreated) vertices.Dispose();
        if (originalPositions.IsCreated) originalPositions.Dispose();
        if (indices.IsCreated) indices.Dispose();
        if (interactionGridA.IsCreated) interactionGridA.Dispose();
        if (interactionGridB.IsCreated) interactionGridB.Dispose();
    }

    void Update()
    {
        //update frame count
        frameCount++;
        //update time
        t += Time.deltaTime * speed;

        MoveCollisionPlane();
        HandleInput();
        GenerateJob();
        UpdateGrid();
        AssignMesh(false);
        SwapGrids();
    }

    public void HandleInput()
    {
        if (Input.GetMouseButton(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit, 1000f))
            {
                //get hit location on plane in 0 - 1 range
                float3 hp = (hit.point - transform.position) / (oceanWidth * meshScale * scale);  
                //transform to grid coord
                int2 gridCoord = (int2)math.abs(hp.xz * interactionGridWidth);
                //for some reason we need to swizzle the xy to yx...? because the gameobject is rotated maybe?
                InteractWithGrid(gridCoord.yx);
            }
        }
    }

    public void SwapGrids()
    {
        if (frameCount % gridUpdateFrequency != 0) return;

        NativeArray<float> temp = interactionGridA;
        interactionGridA = interactionGridB;
        interactionGridB = temp;
    }

    public void InteractWithGrid(int2 position)
    {
        interactionGridA[position.y * interactionGridWidth + position.x] = 1;
    }

    public void UpdateGrid()
    {
        if (frameCount % gridUpdateFrequency != 0) return;

        var jobB = new UpdateInteractionGridRipple
        {
            interactionGridA = interactionGridA,
            interactionGridB = interactionGridB,
            damping = damping,
            width = interactionGridWidth
        }.Schedule((interactionGridWidth - 2) * (interactionGridWidth - 2), 32);
        jobB.Complete();

        //blur the resulting grid otherwise it looks gross
        interactionGridB = GaussianBlur.Blur(interactionGridB, interactionGridWidth, 3, 0.8f);
    }

    public struct UpdateInteractionGridRipple : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float> interactionGridA;
        [NativeDisableParallelForRestriction] public NativeArray<float> interactionGridB;
        [ReadOnly] public int width;
        [ReadOnly] public float damping;
        public void Execute(int index)
        {
            //get x and z values - we skip the edges of the grid so hence the + 1
            int x = (index % width) + 1;
            int z = (index / width) + 1;

            float v0 = interactionGridA[z * width + (x - 1)];
            float v1 = interactionGridA[z * width + (x + 1)];
            float v2 = interactionGridA[(z - 1) * width + x];
            float v3 = interactionGridA[(z + 1) * width + x];

            float val = v0 + v1 + v2 + v3;
            val /= 2;
            val -= interactionGridB[z * width + x];
            val *= damping;
            interactionGridB[z * width + x] = val;
        }
    }

    //move the collision plane to site nicely on top of the ocean
    public void MoveCollisionPlane()
    {
        collisionPlane.transform.localPosition = new float3(oceanWidth / 2, 0, oceanWidth / 2) * meshScale * scale;
        collisionPlane.transform.localScale = new float3(scale);
    }

    public void GenerateJob()
    {
        var job = new WaveJob
        {
            originalPositions = originalPositions,
            vertices = vertices,
            scale = scale,
            time = t,
            width = oceanWidth,
            WaveA = WaveA,
            WaveB = WaveB,
            WaveC = WaveC,
            WaveD = WaveD,
            interactionGrid = interactionGridB,
            interactionGridWidth = interactionGridWidth,
            heightMultiplier = heightMultiplier,
            noiseMultiplier = noiseMultiplier,
            noiseScale = noiseScale,
        }.Schedule(oceanWidth * oceanWidth, 32);
        job.Complete();
    }

    [BurstCompile]
    public struct WaveJob : IJobParallelFor
    {
        [ReadOnly] public int width;
        [ReadOnly] public int interactionGridWidth;
        [ReadOnly] public float noiseMultiplier;
        [ReadOnly] public float noiseScale;
        [ReadOnly] public float scale;
        [ReadOnly] public float time;
        [ReadOnly] public float heightMultiplier;
        [ReadOnly] public GerstnerWaveOptions WaveA;
        [ReadOnly] public GerstnerWaveOptions WaveB;
        [ReadOnly] public GerstnerWaveOptions WaveC;
        [ReadOnly] public GerstnerWaveOptions WaveD;
        [NativeDisableParallelForRestriction, ReadOnly] public NativeArray<float3> originalPositions;
        [NativeDisableParallelForRestriction, ReadOnly] public NativeArray<float> interactionGrid;
        [NativeDisableParallelForRestriction, WriteOnly] public NativeArray<float3> vertices;

        public void Execute(int index)
        {
            int x = index % width;    
            int z = index / width;
            //if is a skirt we can break early
            if (x == 0 || x == width - 1 || z == 0 || z == width - 1) return;
            //get the interactable grid position
            float diff = (float)interactionGridWidth / (float)width;
            int gx = (int)(x * diff);
            int gz = (int)(z * diff);
            //get the interactable grid value
            float m = interactionGrid[gz * interactionGridWidth + gx];
            //get the original vertex position
            float3 p = originalPositions[z * width + x];
            //calculate some noise for an initial y value
            float3 n = new float3(0, FBMNoise(p.xz + time * 0.75f, noiseScale) * noiseMultiplier, 0);
            //add the waves together using FBM
            n += GerstnerWave(p, WaveA, time);
            n += GerstnerWave(p, WaveB, time) * 0.5f;
            n += GerstnerWave(p, WaveC, time) * 0.25f;
            n += GerstnerWave(p, WaveD, time) * 0.2f;
            //add the waves vertex pos to the original vertex position
            float3 newPos = (p + n) * scale;
            //scale the wave height based on whether the water is being
            //interacted with at this point
            newPos.y *= (1.0f - m);
            //lower the water height if it's being interacted with
            //this helps the ripples how up nicer
            newPos.y -= m * 1.5f;
            vertices[z * width + x] = newPos;
        }

        float FBMNoise(float2 pos, float scale)
        {
            float sum = 0;
            float frequency = scale;
            float amp = 0.6f;
            for (int j = 0; j < 6; j++)
            {
                float n = noise.snoise(pos * frequency);
                sum += n * amp;
                frequency *= 2f;
                amp *= 0.5f;
            }
            return sum;
        }

        //taken from https://blog.farazshaikh.com/stories/generating-a-stylized-ocean/
        float3 GerstnerWave(float3 p, GerstnerWaveOptions options, float time)
        {
            float k = 2.0f * math.PI / options.wavelength;
            float c = math.sqrt(9.8f / k);
            float2 d = math.normalize(options.direction);
            float f = k * (math.dot(d, p.xz) - c * time);
            float a = options.steepness / k;

            return new float3(
                d.x * (a * math.cos(f)),
                a * math.sin(f) * heightMultiplier,
                d.y * (a * math.cos(f)));
        }
    }

    public void CreateMesh()
    {
        vCount = oceanWidth * oceanWidth;
        iCount = 0;

        for (int z = 0; z < oceanWidth; z++)
        {
            for (int x = 0; x < oceanWidth; x++)
            {
                //if this is an skirt vertex
                if (x == 0 || x == oceanWidth - 1 || z == 0 || z == oceanWidth - 1) vertices[z * oceanWidth + x] = new float3(x * meshScale, -skirtHeight * meshScale, z * meshScale);
                //otherwise
                else vertices[z * oceanWidth + x] = new float3(x * meshScale, 0, z * meshScale);
                originalPositions[z * oceanWidth + x] = new float3(x * meshScale, 0, z * meshScale);
            }
        }

        //assign the triangles - we only need to do this once
        for (int z = 0; z < oceanWidth - 1; z++)
        {
            for (int x = 0; x < oceanWidth - 1; x++)
            {
                indices[iCount] = (z + 1) * oceanWidth + x;
                indices[iCount + 1] = z * oceanWidth + x + 1;
                indices[iCount + 2] = z * oceanWidth + x;
                indices[iCount + 3] = (z + 1) * oceanWidth + x + 1;
                indices[iCount + 4] = z * oceanWidth + x + 1;
                indices[iCount + 5] = (z + 1) * oceanWidth + x;
                iCount += 6;
            }
        }

        mesh = new Mesh();
        mesh.MarkDynamic();
        filter = GetComponent<MeshFilter>();
        AssignMesh(true);
    }

    public void AssignMesh(bool updateAll)
    {
        //assign vertex data
        NativeArray<VertexAttributeDescriptor> vertexLayout = new NativeArray<VertexAttributeDescriptor>(1, Allocator.Temp);
        vertexLayout[0] = new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3);
        mesh.SetVertexBufferParams(vCount, vertexLayout);
        mesh.SetVertexBufferData<float3>(vertices, 0, 0, vCount);

        //only needed on the initial setup otherwise not required
        if (updateAll)
        {
            //assign indices and submesh
            mesh.SetIndexBufferParams(iCount, IndexFormat.UInt32);
            mesh.SetIndexBufferData<int>(indices, 0, 0, iCount);
            mesh.subMeshCount = 1;
            var subMesh = new SubMeshDescriptor();
            subMesh.indexCount = iCount;
            subMesh.vertexCount = vCount;
            mesh.SetSubMesh(0, subMesh, MeshUpdateFlags.Default);
        }

        //recalculate normals and bounds - is slow
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        filter.mesh = mesh;
    }
}
