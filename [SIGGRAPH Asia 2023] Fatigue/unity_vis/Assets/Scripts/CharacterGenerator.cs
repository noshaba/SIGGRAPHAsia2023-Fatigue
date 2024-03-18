using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CharacterGenerator : MonoBehaviour
{
    public int numberOfCharacter = 10;
    public int frameIntervel = 500;
    public GameObject characterPrefab;
    public BufferedNetworkPoseProvider src;
    private List<GameObject> characterList = new List<GameObject>();
    public Vector3 offset;
    public Vector3 offsetStep;
    // Start is called before the first frame update
    void Start()
    {
        for (int i = 0; i < numberOfCharacter; ++i)
        {
            var character = Instantiate(characterPrefab, new Vector3(0, 0, 0), Quaternion.identity);
            var rr = character.GetComponent<RuntimeRetargetingV3>();
            rr.frameNum = frameIntervel * i;
            rr.src = src;
            rr.offset = new Vector3(offset.x, offset.y, offset.z);
            offset += offsetStep;
            Debug.Log("offset"+offset.ToString());
            characterList.Add(character);
        }
    }

    // Update is called once per frame
    void Update()
    {
        while (characterList.Count > numberOfCharacter)
        {
            var character = characterList[characterList.Count-1];
            Destroy(character);
            characterList.RemoveAt(characterList.Count-1);
        }

        while(characterList.Count < numberOfCharacter)
        {
            var character = Instantiate(characterPrefab, new Vector3(0, 0, 0), Quaternion.identity);
            var rr = character.GetComponent<RuntimeRetargetingV3>();
            rr.src = src;
            rr.offset = new Vector3(offset.x, offset.y, offset.z);
            characterList.Add(character);
            offset += offsetStep;
            Debug.Log("offset"+offset.ToString());
        }

        for (int i = 0; i < numberOfCharacter; ++i)
        {
            var character =characterList[i];
            var rr = character.GetComponent<RuntimeRetargetingV3>();
            rr.frameNum = i * frameIntervel;
        }


    }
}
