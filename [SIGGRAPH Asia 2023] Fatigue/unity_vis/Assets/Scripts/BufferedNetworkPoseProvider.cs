using System.Collections.Generic;
using UnityEngine;
using CustomAnimation;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;



public class BufferedNetworkPoseProvider : CustomAnimationDataBuffer {

    public float scaleFactor;
    [SerializeField]
    public Dictionary<string, int> indexMap;

    virtual public Quaternion GetGlobalRotation(string srcJoint, int i)
    {
        int boneIdx = indexMap[srcJoint];
        int idx = GetActualIndex(i);
        var v = data.poseBuffer[idx].rotations[boneIdx];
        var q = new Quaternion(v.x, v.y, v.z, v.w);
        return q;
    }

    virtual public Vector3 GetGlobalPosition(string srcJoint, int i)
    {
        int boneIdx = indexMap[srcJoint];
        int idx = GetActualIndex(i);
        var v = data.poseBuffer[idx].positions[boneIdx]*scaleFactor;
        return new Vector3(v.x, v.y, v.z);
    }

    virtual public bool GetGlobalPosition(string srcJoint, int i, out Vector3 p)
    {
        p = new Vector3();
        var pose = data.poseBuffer[GetActualIndex(i)];
        if (pose == null) return false;
       int boneIdx = indexMap[srcJoint];
        var v = pose.positions[boneIdx]*scaleFactor;
        p =  new Vector3(v.x, v.y, v.z);
        return true;
    }

    virtual public bool GetGlobalRotation(string srcJoint, int i, out Quaternion q)
    {
        q = new Quaternion();
        var pose = data.poseBuffer[GetActualIndex(i)];
        if (pose == null) return false;
        int boneIdx = indexMap[srcJoint];
        var v = pose.rotations[boneIdx];
        q = new Quaternion(v.x, v.y, v.z, v.w);
        return true;
    }

};
