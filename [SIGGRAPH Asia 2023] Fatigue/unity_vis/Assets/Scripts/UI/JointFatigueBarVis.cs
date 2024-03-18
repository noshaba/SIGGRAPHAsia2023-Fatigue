using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CustomAnimation;
using UnityEngine.UIElements;
using System;

public class JointFatigueBarVis : MonoBehaviour
{
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

    private string[] dofNames = {"x", "y", "z"};

    public BufferedAnimationTCPClient dataSrc;
    public TabbedMenu tabbedMenu;
    public int currentFrame = -1;

    // Update is called once per frame
    void Update()
    {
        if (tabbedMenu == null || dataSrc == null)
            return;
        
        var currentJoint = tabbedMenu.controller.FindSelectedTab()?.name;
        if (currentJoint == null)
            return;
        
        if (!jointOffset.ContainsKey(currentJoint))
            return;
        
        VisualElement tabContent = tabbedMenu.controller.FindSelectedTabContent();

        var jointLength = jointOffset[currentJoint][1];
        var jointStartIdx = jointOffset[currentJoint][0];
        for (int i = 0; i < jointLength; ++i)
        {
            var rc = 100f - GetCurrentValue(currentJoint, i, pose => pose.mf);
            var tl = GetCurrentValue(currentJoint, i, pose => pose.tl);

            var rcBar = GetProgressBar(tabContent, dofNames[i], "RC");
            var tlBar = GetProgressBar(tabContent, dofNames[i], "TL");

            rcBar.value = rc;
            tlBar.value = tl;
        }

    }

    private float GetCurrentValue(string jointname, int i, Func<CKeyframe, float[]> valueFunc)
    {
        var idx = dataSrc.data.GetActualIndex(currentFrame);
        var pose = dataSrc.data.poseBuffer[idx];
        Debug.Assert(i < jointOffset[jointname][1]);
        var value = valueFunc(pose)[jointOffset[jointname][0] + i];
        return value;
    }

    private ProgressBar GetProgressBar(VisualElement tabContent, string dofNmae, string progressBarName)
    {
        return (ProgressBar) tabContent.Q(dofNmae)?.Q(progressBarName);
    }

}
