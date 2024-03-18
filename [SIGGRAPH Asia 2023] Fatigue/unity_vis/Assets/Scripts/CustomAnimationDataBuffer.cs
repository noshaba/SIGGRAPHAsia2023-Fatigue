using System.Collections.Generic;
using UnityEngine;
using CustomAnimation;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

[System.Serializable]
public class BufferData{
    [SerializeField]
    public CKeyframe [] poseBuffer;
    [SerializeField]
    public bool isBufferFull = false;
    [SerializeField]
    public int head = -1;
    [SerializeField]
    public int startIdx = 0;
    [SerializeField]
    public int frameBufferSize = 20000;

    [SerializeField]
    public SkeletonDesc skeletonDesc = null;

    public void AddFrame(CKeyframe pose)
    {
        if (isBufferFull)
        {
            startIdx++;
            startIdx %= frameBufferSize;
        }

        head++;
        if (!isBufferFull && head == frameBufferSize -1)
            isBufferFull = true;
        
        head = head % frameBufferSize;
        poseBuffer[head] = pose;
    }

    public int GetActualIndex(int i)
    {
        // if (i > frameBufferSize)
        // {
        //     Debug.LogError(i.ToString() + " is larger than the buffer size " + frameBufferSize.ToString());
        //     return i;
        // }
        // if (i <= head)
        //     return i;
        // else
        //     return frameBufferSize -i + head;
        var usedBufferSize = GetUsedBufferSize();
        
        if (i < -usedBufferSize || i >= usedBufferSize)
        {
            Debug.LogWarning(i.ToString() + " is larger than the used buffer size " + usedBufferSize.ToString());
        }
        return (usedBufferSize + (startIdx + i) % usedBufferSize) % usedBufferSize;
        // return (frameBufferSize - i + head) % frameBufferSize;

    }

    public int GetUsedBufferSize()
    {
        if (!isBufferFull)
            return head + 1;
        return frameBufferSize;
    }
}


[System.Serializable]
public class BufferData2{
    [SerializeField]
    public BKeyframe [] poseBuffer;
}


public class CustomAnimationDataBuffer : MonoBehaviour {
    public BufferData data = new BufferData();
    public bool Initialized {get => data.poseBuffer[0] != null;}


    virtual public int GetActualIndex(int i)
    {
        return data.GetActualIndex(i);

    }

    virtual public void AddFrame(CKeyframe pose)
    {
        data.AddFrame(pose);
    }

    public BinaryFormatter GetSerializer(){
        //https://answers.unity.com/questions/956047/serialize-quaternion-or-vector3.html
        BinaryFormatter bf = new BinaryFormatter();
        // 1. Construct a SurrogateSelector object
         SurrogateSelector ss = new SurrogateSelector();
         
         Vector3SerializationSurrogate v3ss = new Vector3SerializationSurrogate();
         ss.AddSurrogate(typeof(Vector3), 
                         new StreamingContext(StreamingContextStates.All), 
                         v3ss);
          Vector4SerializationSurrogate v4ss = new Vector4SerializationSurrogate();
         ss.AddSurrogate(typeof(Vector4), 
                         new StreamingContext(StreamingContextStates.All), 
                         v4ss);
         // 2. Have the formatter use our surrogate selector
         bf.SurrogateSelector = ss;
        return bf;
    }


    public void SaveToBinaryFile(string filePath){
        //https://forum.unity.com/threads/saving-game-serialization-binary-resolved.521738/
        
        var bf = GetSerializer();

        FileStream file = File.Open(filePath, FileMode.OpenOrCreate);
        bf.Serialize(file, data);
        file.Close();
    }


    public void LoadFromBinaryFile(string filePath){
        //https://stuartspixelgames.com/2020/07/26/how-to-do-easy-saving-loading-with-binary-unity-c/
        if(File.Exists(filePath))
        {
        // File exists 
            FileStream dataStream = new FileStream(filePath, FileMode.Open);
            var bf = GetSerializer();
            data  = bf.Deserialize(dataStream) as BufferData;

            dataStream.Close();
        }
        else
        {
            // File does not exist
            Debug.LogError("Save file not found in " + filePath);
        }
    }


    public void SaveToFile(string filePath){
        
        string dStr =JsonUtility.ToJson(data, true);
        File.WriteAllText(filePath, dStr);
    }

    public void LoadFromFile(string filePath){

        if(File.Exists(filePath))
        {
        // File exists 
            string dStr = File.ReadAllText(filePath);
            data = JsonUtility.FromJson<BufferData>(dStr);
        }
        else
        {
            // File does not exist
            Debug.LogError("Save file not found in " + filePath);
        }
    }


    public void ReplaceAnimFromFile(string filePath, float scaleFactor, bool switchUpAxis)
    {

        if(File.Exists(filePath))
        {
        // File exists 
            string dStr = File.ReadAllText(filePath);
            Debug.Log("frread data"+ dStr);
            BufferData newData = JsonUtility.FromJson<BufferData>(dStr);
            data.poseBuffer = newData.poseBuffer;
            data.frameBufferSize = newData.poseBuffer.Length;
            data.isBufferFull = true;
            data.head = data.frameBufferSize-1;
            data.startIdx = 0;
            for(int f = 0; f < data.poseBuffer.Length; f++){
                for(int i =0; i <data.poseBuffer[f].positions.Length; i++)
                {
            
                    var p = data.poseBuffer[f].positions[i] * scaleFactor;
                    var r = data.poseBuffer[f].rotations[i];
                    if (switchUpAxis){
                        p = new Vector3(p.x, p.z, p.y);
                        r = new Vector4(r.x, r.z, r.y, -r.w);
                    }
                    data.poseBuffer[f].positions[i] = p;
                    data.poseBuffer[f].rotations[i] = r;
                }
            }
        }
        else
        {
            // File does not exist
            Debug.LogError("Save file not found in " + filePath);
        }
    }


};
