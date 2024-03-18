using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneTransformController : MonoBehaviour
{
    public GameObject camera;
    public float distance = 200;
    public float yPos = -10;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        var rot = transform.rotation.eulerAngles;
        rot.y = Quaternion.LookRotation(-camera.transform.forward, camera.transform.up).eulerAngles.y;
        transform.rotation = Quaternion.Euler(rot.x, rot.y, rot.z);
        // transform.position = camera.transform.position + transform.rotation * new Vector3(0, yPos, -distance);
        transform.position = transform.rotation * new Vector3(camera.transform.position.x, yPos, -distance);
    }
}