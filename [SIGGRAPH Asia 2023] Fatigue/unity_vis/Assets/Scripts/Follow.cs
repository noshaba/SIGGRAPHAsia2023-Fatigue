using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Follow : MonoBehaviour
{

    public Transform following;
    private float replayTime = 0f;
    public float animTime = 0f;
    public float SrcFPS = 30f;
    public int frameNum = 0;
    public bool isPaused = false;
    public Quaternion newRot;
    public Quaternion prevRot;
    
    // Start is called before the first frame update
    void Start()
    {
        Vector3 rot = transform.rotation.eulerAngles;
    	Vector3 followRot = following.rotation.eulerAngles;
    	transform.rotation = Quaternion.Euler(rot.x,rot.y,followRot.y);
    	newRot = transform.rotation;
    	prevRot = newRot;
    }

    // Update is called once per frame
    void Update()
    {
        if (isPaused)
        {
            replayTime = (float)frameNum / SrcFPS;
        }
        else
            replayTime += Time.fixedDeltaTime;
            animTime += Time.fixedDeltaTime;
        frameNum = Mathf.FloorToInt(replayTime * SrcFPS);
    	/*Vector3 rot = transform.rotation.eulerAngles;
    	Vector3 followRot = following.rotation.eulerAngles;
    	transform.rotation = Quaternion.Euler(-270,followRot.y,rot.z);*/
    	
    	transform.position = new Vector3();
    	
    	if(frameNum % 30 == 3) {
    	    Debug.Log("hello");
    	    prevRot = newRot;
    	    Vector3 rot = transform.rotation.eulerAngles;
    	    Vector3 followRot = following.rotation.eulerAngles;
    	    newRot = Quaternion.Euler(rot.x,rot.y,followRot.y-90);
    	    animTime = 0;
    	} 
    	transform.rotation = Quaternion.Lerp(transform.rotation, newRot, animTime);
    	transform.position = new Vector3(following.position.x-0.1f, 2.5f, following.position.z);
    }
}
