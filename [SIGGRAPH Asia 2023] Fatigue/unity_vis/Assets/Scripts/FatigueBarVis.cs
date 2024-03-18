using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using CustomAnimation;

public class FatigueBarVis : MonoBehaviour
{
    public BufferedNetworkPoseProvider dataSrc;
    public ProgressBar avRCBar;
    public ProgressBar avTLBar;
    public int currentFrame = -1;

    private void OnEnable()
    {
        var rootVisualElement = GetComponent<UIDocument>().rootVisualElement;
        avRCBar = rootVisualElement.Q<ProgressBar>("Average-RC");
        avTLBar = rootVisualElement.Q<ProgressBar>("Average-TL");
        
    }
    // Start is called before the first frame update
    void Start()
    {
       
    }

    // Update is called once per frame
    void Update()
    {
        if (dataSrc.data.head < 0)
            return;
        var idx = dataSrc.data.GetActualIndex(currentFrame);
        
        if (dataSrc.data.poseBuffer[idx] != null)
        {
            var pose = dataSrc.data.poseBuffer[idx];
            VisBar(avRCBar, 100.0f - Average(pose.mf));
            VisBar(avTLBar, Average(pose.tl));
        }
    }

    public void VisBar(ProgressBar pb, float val)
    {
        pb.value = val;
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
