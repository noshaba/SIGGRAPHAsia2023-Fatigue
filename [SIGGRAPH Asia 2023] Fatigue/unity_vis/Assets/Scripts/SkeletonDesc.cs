using UnityEngine;

namespace CustomAnimation
{
	[System.Serializable]
	public class SkeletonDesc
	{
    	[SerializeField]
		public string root;

    	[SerializeField]
		public JointDesc[] jointDescs;

    	[SerializeField]
		public string[] jointSequence;
	}



}
