using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using CustomAnimation;

[CustomEditor(typeof(CustomAnimationPlayer))]
public class CustomAnimationPlayerEditor : Editor
{

    public override void OnInspectorGUI()
    {
        // will enable the default inpector UI 
        base.OnInspectorGUI();
        CustomAnimationPlayer poseBuffer = (CustomAnimationPlayer)target;
     
        if (GUILayout.Button("Load Animation"))
        {
            poseBuffer.LoadAnimation();
        }
        if (GUILayout.Button("Replace Animation"))
        {
            poseBuffer.ReplaceAnimation();
        }
    }

}

