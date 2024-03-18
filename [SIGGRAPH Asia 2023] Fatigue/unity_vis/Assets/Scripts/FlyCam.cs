using UnityEngine;
using UnityEngine.Internal;


public class FlyCam : MonoBehaviour
{
    public float camHeight = 1;
    public float distance = 5;
    /// <summary>The target which camera should look at </summary>
    public Transform target;
    
    ///<summary>Translation in global coordinate frame</summary>
    public Vector3 translation = Vector3.zero;
    public bool lockRelativeRotation = false;
    public bool showIsaacSide = false;
    /// <summary>Camera rotation in target coordinate frame</summary>
    private Quaternion localRotation;
    /// <summary>Camera rotation in global coordinate frame</summary>
    private Quaternion globalRotation;
    public Vector3 rotation
    {
        get { return globalRotation.eulerAngles; }
        set
        {
            globalRotation = Quaternion.Euler(value);
            localRotation = Quaternion.Inverse(target.transform.rotation) * globalRotation;
        }
    }

    void Start()
    {
        if(!showIsaacSide)
            LookAtTargetFrom(Vector3.forward, Vector3.up, distance);
    }
    void Update()
    {
        if (!lockRelativeRotation)
        {
            transform.rotation = globalRotation;
            localRotation = Quaternion.Inverse(target.transform.rotation) * globalRotation;
        }
        else
        {
            //transform.rotation = target.transform.rotation * localRotation;
        }
        
        transform.position = target.position - distance * transform.forward + translation + (camHeight - target.position.y) * transform.up;
    }

    public void ClearTranslation()
    {
        translation = Vector3.zero;
    }

    /// <summary>Look at target from a point</summary>
    /// <param name="targetLocalDirection">target direction from camera position in target frame</param>
    /// <param name="targetLocalCameraUp">camera up direction in target frame</param>
    /// <param name="distance">distance between camera and target</param>
    public void LookAtTargetFromLocal(Vector3 targetLocalDirection, Vector3 targetLocalCameraUp, float distance)
    {
        Vector3 direction = target.TransformDirection(targetLocalDirection);
        Vector3 cameraUp = target.TransformDirection(targetLocalCameraUp);
        LookAtTargetFrom(direction, cameraUp, distance);
    }

    /// <summary>Look at target from a point</summary>
    /// <param name="targetLocalDirection">target direction from camera position in global frame</param>
    /// <param name="targetLocalCameraUp">camera up direction in global frame</param>
    /// <param name="distance">distance between camera and target</param>
    public void LookAtTargetFrom(Vector3 direction, Vector3 cameraUp, float distance)
    {
        direction = direction.normalized;
        cameraUp = cameraUp.normalized;
        Vector3 cameraForward = -direction;
        ClearTranslation();
        rotation = (Quaternion.LookRotation(cameraForward, cameraUp)).eulerAngles;
        this.distance = distance;
    }


}
