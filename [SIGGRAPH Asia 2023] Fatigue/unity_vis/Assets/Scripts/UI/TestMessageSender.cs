using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CustomAnimation;
using UnityEngine.UIElements;

public class TestMessageSender : MonoBehaviour
{
    public bool sendingData = true;
    public BufferedAnimationTCPClient client;
    private Slider fatigueFSlider;
    private Slider fatigueRSlider;
    private Slider fatigueSmallRSlider;
    private Dictionary<string, string> veDataNameMapping;
    public float F;
    public float R;
    public float samllR;
    Dictionary<string, float> data = new Dictionary<string, float>();
    // Start is called before the first frame update
    void Start()
    {
        fatigueFSlider.value = F;
        fatigueRSlider.value = R;
        fatigueSmallRSlider.value = samllR;
        if (sendingData)
        {
            sendData(veDataNameMapping[fatigueFSlider.name], fatigueFSlider.value);
            sendData(veDataNameMapping[fatigueRSlider.name], fatigueRSlider.value);
            sendData(veDataNameMapping[fatigueSmallRSlider.name], fatigueSmallRSlider.value);
        }
        
    }

    private void OnEnable()
    {
        var rootVisualElement = GetComponent<UIDocument>().rootVisualElement;
        veDataNameMapping = new Dictionary<string, string>();

        fatigueFSlider = rootVisualElement.Q<Slider>("FatigueF-Slider");
        fatigueFSlider.RegisterValueChangedCallback(OnFloatValueChanged);
        veDataNameMapping.Add(fatigueFSlider.name, "fatigueF");
        fatigueFSlider.label = "F: " + fatigueFSlider.value.ToString();

        fatigueRSlider = rootVisualElement.Q<Slider>("FatigueR-Slider");
        fatigueRSlider.RegisterValueChangedCallback(OnFloatValueChanged);
        veDataNameMapping.Add(fatigueRSlider.name, "fatigueR");
        fatigueRSlider.label = "R/F: " + fatigueRSlider.value.ToString();

        fatigueSmallRSlider = rootVisualElement.Q<Slider>("Fatigue_r-Slider");
        fatigueSmallRSlider.RegisterValueChangedCallback(OnFloatValueChanged);
        veDataNameMapping.Add(fatigueSmallRSlider.name, "fatigue_r");
        fatigueSmallRSlider.label = "r: " + fatigueSmallRSlider.value.ToString();


    }

    // Update is called once per frame
    void Update()
    {
        F = fatigueFSlider.value;
        R = fatigueRSlider.value;
        samllR = fatigueSmallRSlider.value;
    }

    private void OnFloatValueChanged(ChangeEvent<float> evt)
    {
        var visaulElement = (VisualElement)evt.currentTarget;
        Slider slider = (Slider)visaulElement;
        string[] words = slider.label.Split(":");
        slider.label = words[0] + ": " + evt.newValue.ToString();
        if (sendingData)
            sendData(veDataNameMapping[visaulElement.name], evt.newValue);
    }

    private void sendData(string key, float val)
    {
        // Debug.LogWarning("sending data");
        string jsonData = "{\"" + key + "\":" + val.ToString() + "}";
        client.messageQueue.Add(jsonData);
    }
}
