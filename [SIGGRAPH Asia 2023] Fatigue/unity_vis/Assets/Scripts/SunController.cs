using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SunController : MonoBehaviour
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
        // sun position
        transform.position = new Vector3(camera.transform.position.x, yPos, distance);
    }
}
