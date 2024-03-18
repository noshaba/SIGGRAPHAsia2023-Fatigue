
namespace CustomAnimation {
	[System.Serializable]
	public class JointDesc
	{
		public string name; // name in server
		public string targetName; // name in Unity
		public string[] children;
	}
}
