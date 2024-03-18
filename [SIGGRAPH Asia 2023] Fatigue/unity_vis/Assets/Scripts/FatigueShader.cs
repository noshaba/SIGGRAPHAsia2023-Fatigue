using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FatigueShader : MonoBehaviour
{
    public Color fatigue;
    public Color noFatigue;
    public float fatigueCut = 0.5f;
    public float restCut = 0.4f;
    public ReplayController replayController;
    public BufferedNetworkPoseProvider dataSrc;
    //public ReplayController replayController;
    public Material mat;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        if (dataSrc.data.head < 0)
            return;
        var currentFrame = -1;
        if (replayController)
            currentFrame = replayController.frameNum;
        var idx = dataSrc.data.GetActualIndex(currentFrame);
        float MF_a = 0f;
        
        if (dataSrc.data.poseBuffer[idx] != null)
        {
            var pose = dataSrc.data.poseBuffer[idx];
            MF_a = Average(pose.mf) / 100.0f;
        }

        //float grad = (Mathf.Clamp(MF_a, restCut, fatigueCut) - restCut) / (fatigueCut - restCut);

        float grad = (Mathf.Clamp(MF_a, restCut, fatigueCut) - restCut) / (fatigueCut - restCut);

        Color col = Color.Lerp(noFatigue, fatigue, grad);
        mat.SetColor("_EmissionColor", col);
    }

    float Average(float[] vals)
    {
        if (vals.Length == 0)
            return 0.0f;
        float average = 0;
        for (int i = 0; i < vals.Length; ++i)
            average += vals[i];
        average /= (float)vals.Length;
        return average;
    }
}
