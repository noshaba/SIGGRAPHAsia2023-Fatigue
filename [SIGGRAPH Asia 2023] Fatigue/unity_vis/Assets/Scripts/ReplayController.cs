using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReplayController : MonoBehaviour
{
    public float SrcFPS = 30f;
    public float maxFrame = -1f;
    public float maxReplayTime = -1f;
    public bool isPaused = false;
    public int frameNum = 0;
    private float replayTime = 0f;
    public RuntimeRetargetingV3 runtimeRetargetingV3;
    public FatigueBarVis fatigueBarVis;
    public FatiguePlot fatiguePlot;
    public JointFatigueBarVis jointFatigueBarVis;
    public BufferedNetworkPoseProvider dataSrc;


    // Update is called once per frame
    void Update()
    {
        if (isPaused)
        {
            replayTime = (float)frameNum / SrcFPS;
        }
        else
            replayTime += Time.deltaTime;
        if (maxReplayTime > 0f && replayTime > maxReplayTime)
            replayTime = 0;
        frameNum = Mathf.FloorToInt(replayTime * SrcFPS);
        if (runtimeRetargetingV3)
            runtimeRetargetingV3.frameNum = frameNum;
        if (fatigueBarVis)
            fatigueBarVis.currentFrame = frameNum;
        if (fatiguePlot)
            fatiguePlot.currentFrame = frameNum;
        if (jointFatigueBarVis)
            jointFatigueBarVis.currentFrame = frameNum;

        if(frameNum == dataSrc.data.head)
            UnityEditor.EditorApplication.isPlaying = false;
        //    isPaused = true;
    }


}
