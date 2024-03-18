using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CustomAnimation;
using System.IO;

public class CustomAnimationPlayer : BufferedNetworkPoseProvider
{
    public string filename = "";
    public string clipfilename = "";
    public bool paused;
    Dictionary<string, GameObject> points;
    protected bool initialized;
    public float visScaleFactor;
    public bool createVisBodies = true;
    public int frameIdx = 0;
    public bool switchUpAxis= true;

    // Start is called before the first frame update
    void Start()
    {
        points = new Dictionary<string, GameObject>();
    }

    public void Update()
    {

        if (paused) return;
        CKeyframe pose = null;

        if (data.head >= 0){
            if (frameIdx > data.head) frameIdx =0;
            pose = data.poseBuffer[frameIdx];
            frameIdx += 1;
        }

        if (initialized && pose != null)
        {
            int index = 0;
            foreach (var name in data.skeletonDesc.jointSequence)
            {
                
                var r = pose.rotations[index];
                points[name].transform.position =  pose.positions[index];
                points[name].transform.rotation = new Quaternion(r.x, r.y, r.z, r.w);

                index += 1;
            }
        }
        else if (!initialized && data.skeletonDesc != null)
        {
            initSkeleton();
        }


    }

    public void initSkeleton()
    {
        if (data.skeletonDesc != null)
        {
            
            Debug.Log("generated from skeleton desc");
            createDebugSkeleton(data.skeletonDesc);
            
            buildPoseParameterIndexMap(data.skeletonDesc.jointSequence);
            initialized = true;
        }
        Debug.Log("processed skeleton");
    }


    public void LoadAnimation()
    {
        var _filename = Path.Combine(Application.streamingAssetsPath, filename);
        base.LoadFromBinaryFile(_filename);
    }

    protected void buildPoseParameterIndexMap(string[] jointSequence)
    {

        indexMap = new Dictionary<string, int>();
        foreach (string name in jointSequence)
        {
            int idx = Array.IndexOf(jointSequence, name);
            indexMap [name] = idx;
        }
    }

    public void createDebugSkeleton(SkeletonDesc skeletonDesc)
    {
        Debug.Log("createDebugSkeleton");
        foreach (var jointName in data.skeletonDesc.jointSequence)
        {

            if (createVisBodies){
                points[jointName] = createJointVisualization(visScaleFactor);
            }else{
                points[jointName] = new GameObject();
            }
            points[jointName].name = jointName;
            points[jointName].transform.parent = transform;
        }
    }

    protected GameObject createJointVisualization(float scale=1.0f)
    {
        GameObject jointObj = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        Mesh mesh = jointObj.GetComponent<MeshFilter>().mesh;
        List<Vector3> newVertices = new List<Vector3>();
        var r = Quaternion.Euler(90, 0, 0);
        for (int i = 0; i < mesh.vertices.Length; i++)
        {
            var v = mesh.vertices[i];
            var newV =  new Vector3(v.x * scale, scale * v.y, v.z * scale);
            newVertices.Add(newV);
        }
        jointObj.GetComponent<MeshFilter>().mesh.vertices = newVertices.ToArray();
        jointObj.GetComponent<MeshFilter>().mesh.RecalculateNormals();

        return jointObj;
    }

    public void ReplaceAnimation()
    {
        var _filename = Path.Combine(Application.streamingAssetsPath, clipfilename);
        base.ReplaceAnimFromFile(_filename, scaleFactor, switchUpAxis);
    }

}
