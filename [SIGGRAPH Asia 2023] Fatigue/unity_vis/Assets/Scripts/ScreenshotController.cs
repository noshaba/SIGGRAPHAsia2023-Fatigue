//src: https://answers.unity.com/questions/22954/how-to-save-a-picture-take-screenshot-from-a-camer.html
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScreenshotController : MonoBehaviour
{
    public int resWidth = 2550;
    public int resHeight = 3300;

    bool queueScreenShot = false;
    public Camera camera = null;
    public string prefix = "";

    void Start()
    {
        queueScreenShot = false;
    }

    public static string ScreenShotName(string prefix)
    {
        return string.Format("{0}/{1}_{2}.png",
                             Application.dataPath,
                             prefix,
                             System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }

    public void TakeScreenshot()
    {
        queueScreenShot = true;
    }

    void LateUpdate()
    {
        queueScreenShot |= Input.GetKeyDown("k");
        if (queueScreenShot)
        {
            saveScreenShot();
        }
    }

    void saveScreenShot()
    {
        if (camera == null) return;
        RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
        camera.targetTexture = rt;
        Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        camera.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
        camera.targetTexture = null;
        RenderTexture.active = null; // JC: added to avoid errors
        Destroy(rt);
        byte[] bytes = screenShot.EncodeToPNG();
        string filename = ScreenShotName(prefix);
        System.IO.File.WriteAllBytes(filename, bytes);
        Debug.Log(string.Format("Took screenshot to: {0}", filename));
        queueScreenShot = false;
    }
}