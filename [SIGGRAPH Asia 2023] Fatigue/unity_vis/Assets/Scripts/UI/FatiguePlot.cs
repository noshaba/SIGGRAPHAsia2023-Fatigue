using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CustomAnimation;
using System;

public class FatiguePlot : MonoBehaviour
{
	public float YMin = 0f;
	public float YMax = 100f;

    public float xPos = 0.5f;
    public float yPos = 0.25f;

    public BufferedNetworkPoseProvider dataSrc;
    public TabbedMenu tabbedMenu;
    public int bufferSize = 2000;
    public int currentFrame = -1;
    public string currentJoint="";
    public bool isReplay = false;
    public bool showTL = true;
    public bool showMA = true;
    public bool showMR = true;
    public bool showMF = true;
    public bool showRC = true;
    public bool showAT = true;

    private Queue<float> valueBuffer = new Queue<float>();
    private List<CKeyframe> poseBuffer = new List<CKeyframe>();
    private Dictionary<string, int[]> jointOffset = new Dictionary<string, int[]> {
        {"AbdomenTab", new int[] {0, 3}},
        {"NeckTab", new int[] {3, 3}},
        {"RightShoulderTab", new int[] {6, 3}},
        {"RightElbowTab", new int[] {9, 1}},
        {"LeftShoulderTab", new int[] {10, 3}},
        {"LeftElbowTab", new int[] {13, 1}},
        {"RightHipTab", new int[] {14, 3}},
        {"RightKneeTab", new int[] {17, 1}},
        {"RightAnkleTab", new int[] {18, 3}},
        {"LeftHipTab", new int[] {21, 3}},
        {"LeftKneeTab", new int[] {24, 1}},
        {"LeftAnkleTab", new int[] {25, 3}}
    };
    

    void FillPoseBuffer()
    {
        // Debug.Assert(bufferSize < dataSrc.data.poseBuffer.Length);
        if (dataSrc.data.head < 0)
            return;
        poseBuffer.Clear();
        var dataLength = Math.Min(bufferSize, dataSrc.data.GetUsedBufferSize());
        if (isReplay)
            dataLength = Math.Min(bufferSize, currentFrame+1);
        for (int i = currentFrame - dataLength + 1; i <= currentFrame; ++i)
        {
            var idx = dataSrc.GetActualIndex(i);
            if (idx == dataSrc.data.startIdx)
                poseBuffer.Clear();
            var pose = dataSrc.data.poseBuffer[idx];
            if(pose.reset)
                poseBuffer.Clear();
            // Debug.Log(idx);
            if (pose != null)
            {
                poseBuffer.Add(pose);
            }
        }
    }

    float[] GetDrawingData(string jointName, Func<CKeyframe, float[]> valueFunc)
    {
        if (!jointOffset.ContainsKey(jointName) || poseBuffer.Count == 0)
            return null;
        var jointStartIdx = jointOffset[jointName][0];
        var jointLength = jointOffset[jointName][1];
        float[] values = new float[jointLength];
        float[] averageVals = new float[poseBuffer.Count];
        int i = 0;
        foreach (CKeyframe pose in poseBuffer)
        {
            Array.Copy(valueFunc(pose), jointStartIdx, values, 0, jointLength);
            var averageVal = Average(values);
            averageVals[i] = averageVal;
            i++;
        }
        return averageVals;
    }

	// void OnDrawGizmos() {
	// 	if(!Application.isPlaying) {
	// 		Draw();
	// 	}
	// }

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
        if (tabbedMenu == null)
            return;
        currentJoint = tabbedMenu.controller.FindSelectedTab()?.name;
        FillPoseBuffer();
        if (!jointOffset.ContainsKey(currentJoint))
            return;
        
        float[] tl = GetDrawingData(currentJoint, pose => pose.tl);
        float[] ma = GetDrawingData(currentJoint, pose => pose.ma);
        float[] mr = GetDrawingData(currentJoint, pose => pose.mr);
        float[] mf = GetDrawingData(currentJoint, pose => pose.mf);
        float[] rc = new float[mf.Length];
        float[] at = new float[tl.Length];

        for(int i = 0; i < rc.Length; i++) 
        {
            rc[i] = 100f - mf[i];
            at[i] = (rc[i] > tl[i]) ? tl[i] : rc[i];
        }

        List<float[]> valList = new List<float[]>();
        List<Color> colorList = new List<Color>();
        if (showTL)
        {
            valList.Add(tl);
            colorList.Add(new Color(0.8f, 0.6f, 0f));
        }
        if (showMR)
        {
            valList.Add(mr);
            colorList.Add(new Color(0f, 0.9f, 1f));
        }
        if (showMF)
        {    
            valList.Add(mf);
            colorList.Add(new Color(229f/255f, 18f/255f, 111f/255f));
        }
        if (showRC)
        {
            valList.Add(rc);
            colorList.Add(new Color(0f, 1f, 165f/255f));
        }
        if (showMA)
        {
            valList.Add(ma);
            colorList.Add(new Color(250f/255f, 180f/255f, 1f));
        }
        if(showAT)
        {
            valList.Add(at);
            colorList.Add(new Color(1f, 1f, 1f));
        }
        foreach(float[] vals in valList)
            if (vals.Length < 2)
                return;
        
        // Color[] lineColors = {new Color(1.0f, 0.671f, 0.016f), new Color(0.709f, 0.505f, 0.6f), new Color(0.239f, 0.521f, 0.776f), new Color(0.611f, 0.137f, 0.333f)};
        Color[] lineColors = colorList.ToArray();

		UltiDraw.Begin();

        // UltiDraw.DrawGUIFunctions(new Vector2(0.5f, 0.11f), new Vector2(0.98f, 0.2f), valList, YMin, YMax, 0.002f, new Color(0.75f, 0.75f, 0.75f, 0.5f), lineColors);
        UltiDraw.DrawGUIFunctionsCustomize(new Vector2(xPos, yPos), new Vector2(0.98f, 0.2f), bufferSize, valList, YMin, YMax, 0.002f, new Color(0.1f, 0.1f, 0.1f, 0.75f), lineColors, 0.002f, new Color(1f, 1f, 1f, 1f));

		UltiDraw.End();
        
	}

    public float Average(float[] vals)
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
