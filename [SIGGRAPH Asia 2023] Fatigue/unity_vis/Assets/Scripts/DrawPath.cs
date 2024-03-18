using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawPath : MonoBehaviour
{
    public Color color = Color.red;
    public ReplayController replayController;
    public float width = 0.1f;
    public int startFrame = 0;
    public bool drawOnGround = false;
    List<Vector3> jointPos = new List<Vector3>();
    LineRenderer lineRenderer;
    public BufferedNetworkPoseProvider dataSrc;
    int frame = 0;
    bool sceneLoaded = false;
    
    // Start is called before the first frame update
    void Start()
    {
        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.material = new Material(Shader.Find("Legacy Shaders/Particles/Additive"));
        lineRenderer.SetColors(color, color);
        lineRenderer.SetWidth(width, width);
    }

    // Update is called once per frame
    void Update()
    {
        var currentFrame = -1;
        if (replayController)
            currentFrame = replayController.frameNum;
        frame = dataSrc.data.GetActualIndex(currentFrame);
        if(frame > startFrame) 
        {
            jointPos.Add(transform.position);

            lineRenderer.SetVertexCount(jointPos.Count);

            for (int i = 0; i < jointPos.Count; i++)
            {
            	Vector3 pos = jointPos[i];
            	if(drawOnGround)
            	    pos = new Vector3(jointPos[i].x, 0, jointPos[i].z);
            	    
                lineRenderer.SetPosition(i, pos);
            }
        }
    }
}
