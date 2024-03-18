using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class RuntimeRetargetingV3 : MonoBehaviour
{
    [Serializable]
    public class RetargetingMap
    {
        public string src;
        public string dst;
        public Vector3 srcUp;
        public Vector3 srcSide;
        public Vector3 dstUp;
        public Vector3 dstSide;
        public Transform dstT;
        public bool active = true;

    }
    public BufferedNetworkPoseProvider src;
    public List<RetargetingMap> retargetingMap;
    public float visLength = 10;
    public string rootName;
    public Vector3 rootOffset;
    public bool bringToLocal = true;
    public bool mirror = false;
    public bool showSrcGizmos = false;
    public bool showDstGizmos = false;
    bool initialized = false;
    Animator anim;
    public bool automatic = false;
    public float translationScale = 1;
    public int frameNum = -1;
    public Vector3 offset;

    // Start is called before the first frame update
    void Start()
    {
        anim = GetComponent<Animator>();
        SetTransforms();
    }

    public void SetTransforms()
    {
        
       var  _transforms = GetComponentsInChildren<Transform>().ToList();

        for (int i =0;i <retargetingMap.Count; i++)
        {
            var dstT = _transforms.First(x => x.name == retargetingMap[i].dst);
            retargetingMap[i].dstT = dstT;
        }
        initialized = true;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (initialized && automatic) OrientToTarget();
    }

    public void TriggerUpdate()
    {
        if (initialized)
        {
            OrientToTarget();
        }
    }

    public Quaternion RetargetRotation(Quaternion srcRotation, RetargetingMap m)
    {
        var globalSrcUp = srcRotation * m.srcUp;
        globalSrcUp = globalSrcUp.normalized;
        var globalSrcSide = srcRotation * m.srcSide;
        globalSrcSide = globalSrcSide.normalized;
        var rotation1 = Quaternion.FromToRotation(m.dstUp, globalSrcUp).normalized;
        var dstSide = rotation1 * m.dstSide;
        var rotation2 = Quaternion.FromToRotation(dstSide.normalized, globalSrcSide).normalized;
        var rotation = Quaternion.Normalize(rotation2 * rotation1);
        var dstUp = rotation * m.dstUp;
        var rotation3 = Quaternion.FromToRotation(dstUp.normalized, globalSrcUp).normalized;
        rotation = Quaternion.Normalize(rotation3 * rotation);
        return rotation;
    }

    public void OrientToTarget()
    {
        Quaternion srcRotation;
        Vector3 srcPosition;
        if (src.data==null)return;
        if (src.indexMap==null)return;
        foreach (var m in retargetingMap)
        {
            if (! m.active) continue;

            var dstT = m.dstT;
            if (!src.GetGlobalRotation(m.src, frameNum, out srcRotation))
            {
                //Debug.Log(m.src.ToString() + "does not exist");
                continue;
            }
            var rotation = RetargetRotation(srcRotation, m);
            if (m.dst == rootName)
            {
                dstT.rotation = rotation.normalized;
                src.GetGlobalPosition(m.src, frameNum, out srcPosition);
                dstT.position = offset+srcPosition*translationScale;
                // mirror
                if(mirror)
                    dstT.position = new Vector3(-dstT.position.x, dstT.position.y, dstT.position.z);
            }
            else
            {
                var invRot = Quaternion.identity;
                if (dstT.parent != null)
                {
                    invRot = Quaternion.Inverse(dstT.parent.rotation);
                }
                dstT.localRotation = invRot * rotation.normalized;
            }

            
        }
        
    }

    void OnDrawGizmos()
    {
        if (!showSrcGizmos && !showDstGizmos) return;
        foreach (var m in retargetingMap)
        {
            if (m.dstT != null)
            {
                if (showSrcGizmos) { 
                    Quaternion srcRotation;
                    if (src.GetGlobalRotation(m.src, frameNum, out srcRotation))
                    {
                        var srcPosition = src.GetGlobalPosition(m.src, frameNum);
                        var globalSrcUp = srcRotation * m.srcUp * visLength;
                        var globalSrcSide = srcRotation * m.srcSide * visLength;
                        Debug.DrawLine(srcPosition, srcPosition + globalSrcUp, Color.green);
                        Debug.DrawLine(srcPosition, srcPosition + globalSrcSide, Color.red);
                    }
                    
                }
                if (showDstGizmos) { 
                    var globalDstUp = m.dstT.rotation * m.dstUp * visLength;
                    var globalDstSide = m.dstT.rotation * m.dstSide * visLength;
                    Debug.DrawLine(m.dstT.position, m.dstT.position + globalDstUp, Color.green);
                    Debug.DrawLine(m.dstT.position, m.dstT.position + globalDstSide, Color.red);
                }
            }
        }
    }
}
