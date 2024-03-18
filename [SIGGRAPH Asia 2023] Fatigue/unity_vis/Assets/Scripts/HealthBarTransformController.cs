using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HealthBarTransformController : MonoBehaviour
{
    public Transform CharacterHip;
    public GameObject cam;
    public float barHeight = 2;
    private RectTransform rectTransform;
    // Start is called before the first frame update
    void Start()
    {
        rectTransform = GetComponent<RectTransform>();
    }

    // Update is called once per frame
    void Update()
    {
        rectTransform.position = CharacterHip.position + new Vector3(0, barHeight, 0);
        rectTransform.rotation = cam.transform.rotation;
    }
}
