using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using CustomAnimation;

[CustomEditor(typeof(BufferedAnimationTCPClient))]
public class BufferedAnimationTCPClientEditor : Editor
{

    public override void OnInspectorGUI()
    {
        // will enable the default inpector UI 
        base.OnInspectorGUI();
        BufferedAnimationTCPClient poseBuffer = (BufferedAnimationTCPClient)target;
     
        if (GUILayout.Button("Save Animation"))
        {
            poseBuffer.SaveAnimation();
        }
    }

}

