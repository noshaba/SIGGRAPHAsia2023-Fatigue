using UnityEngine;

[RequireComponent(typeof(FlyCam))]
public class FlyCamController : MonoBehaviour
{
    private FlyCam flyCam;
    private Camera camera;
    public float rotationSensibility = 0.5F;
    public float zoomSensiblility = 0.1F;
    private bool isMousePressing;
    private Vector3 currentMousePos;
    private Vector3 lastMousePos;
    private Vector3 deltaRotation;

    private bool isCtrlMousePressing;
    private Vector3 deltaTranslation;

    void Awake()
    {
        flyCam = GetComponent<FlyCam>();
        camera = GetComponent<Camera>();
        isMousePressing = false;
    }
    void Update()
    {
        RotateCam();
        ChangeDistance();
        TranslateCam();
        Togglelocklocal();
    }

    // move camera around traget using middle mouse button
    void RotateCam()
    {       
        if(Input.GetMouseButton(1))
        {
            currentMousePos = Input.mousePosition;
            if (!isMousePressing)
                isMousePressing = true;
            else
            {
                deltaRotation = (currentMousePos - lastMousePos) * rotationSensibility;
                deltaRotation = new Vector3(-deltaRotation.y, deltaRotation.x, 0);
                flyCam.rotation += deltaRotation;
            }
            lastMousePos = currentMousePos;
        }
        else
        {
            isMousePressing = false;
        }
    }

    void ChangeDistance()
    {
        flyCam.distance -= Input.mouseScrollDelta.y * zoomSensiblility;
    }

    void TranslateCam()
    {
        if(Input.GetMouseButton(2))
        {
            currentMousePos = Input.mousePosition;
            if (!isCtrlMousePressing)
                isCtrlMousePressing = true;
            else
            {
                Vector3 currentPos = new Vector3(currentMousePos.x, currentMousePos.y, flyCam.distance);
                Vector3 lastPos = new Vector3(lastMousePos.x, lastMousePos.y, flyCam.distance);
                deltaTranslation = camera.ScreenToWorldPoint(currentPos) - camera.ScreenToWorldPoint(lastPos);
                deltaTranslation = -deltaTranslation;
                flyCam.translation += deltaTranslation;
            }
            lastMousePos = currentMousePos;
        }
        else
        {
            isCtrlMousePressing = false;
        }
    }

    void Togglelocklocal()
    {
        if (Input.GetKeyUp(KeyCode.F))
        {
            flyCam.lockRelativeRotation = !flyCam.lockRelativeRotation;
        }
    }
}