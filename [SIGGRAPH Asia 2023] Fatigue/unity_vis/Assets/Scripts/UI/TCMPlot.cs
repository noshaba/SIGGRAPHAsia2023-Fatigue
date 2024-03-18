using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TCMPlot : MonoBehaviour
{
    public TestMessageSender msgSender;
    public float tl = 50f;
    public int length = 100;

    public float xPos = 0.12f;
    public float yPos = 0.6f;
    public float width = 0.2f;
    public float height = 0.16f;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {

    }

    List<float[]> GenTCMData(float F, float R, float smallR, float tl, int length)
    {
        float[] TL_hist = new float[length];
        float[] MA_hist = new float[length];
        float[] MF_hist = new float[length];
        float[] MR_hist = Enumerable.Repeat(100f, length).ToArray();
        float[] RC_hist = Enumerable.Repeat(100f, length).ToArray();

        // init. target load (box function)
        for(int i = length / 4; i < length / 4 * 3; i++)
            TL_hist[i] = tl;

        float Ct = 0f;
        float rR = 0f;

        R *= F;

        // 3CC model
        for(int t = 0; t < length - 1; t++) {
            float TL = TL_hist[t];
            float MA = MA_hist[t];
            float MR = MR_hist[t];
            float MF = MF_hist[t];

            rR = R;

            if((MA < TL) && (MR > (TL - MA)))
                Ct = 10 * (TL - MA);
            else if((MA < TL) && (MR < (TL - MA)))
                Ct = 10 * MR;
            else if(MA >= TL) {
                Ct = 10 * (TL - MA);
                rR = smallR * R;
            }
            
            MF_hist[t + 1] = MF + Time.fixedDeltaTime * (F * MA - rR * MF);
            MR_hist[t + 1] = MR + Time.fixedDeltaTime * (-Ct + rR * MF);
            MA_hist[t + 1] = 100 - MF_hist[t + 1] - MR_hist[t + 1];
            RC_hist[t + 1] = 100 - MF;
        }

        return new List<float[]>{ TL_hist, MR_hist, MF_hist, RC_hist, MA_hist};
    }

    void OnRenderObject() {
		Draw(msgSender.F, msgSender.R, msgSender.samllR, tl, length);
	}

    void Draw (float F, float R, float smallR, float tl, int length)
    {
        List<float[]> valList = GenTCMData(F, R, smallR, tl, length);
        List<Color> colorList = new List<Color>();
        colorList.Add(new Color(0.8f, 0.6f, 0f));                   // TL
        colorList.Add(new Color(0f, 0.9f, 1f));                     // MR
        colorList.Add(new Color(229f/255f, 18f/255f, 111f/255f));   // MF
        colorList.Add(new Color(0f, 1f, 165f/255f));                // RC
        colorList.Add(new Color(250f/255f, 180f/255f, 1f));         // MA

        Color[] lineColors = colorList.ToArray();

        UltiDraw.Begin();
        UltiDraw.DrawGUIFunctions(new Vector2(xPos, yPos), new Vector2(width, height), valList, 0, 100, 0.002f, new Color(0.1f, 0.1f, 0.1f, 0.75f), lineColors, 0.002f, new Color(1f, 1f, 1f, 1f));
        UltiDraw.End();
    }
}
