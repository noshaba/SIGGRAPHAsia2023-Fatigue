using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class HealthBarVis : MonoBehaviour
{
    public BufferedNetworkPoseProvider dataSrc;
    public ReplayController replayController;
    public Image image;
    // Start is called before the first frame update
    void Start()
    {
        image = GetComponent<Image>();
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
        if (dataSrc.data.poseBuffer[idx] != null)
        {
            var pose = dataSrc.data.poseBuffer[idx];
            image.fillAmount = 1f - Average(pose.mf) / 100.0f;
        }
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