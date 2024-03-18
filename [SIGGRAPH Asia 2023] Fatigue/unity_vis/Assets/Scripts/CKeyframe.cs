//https://answers.unity.com/questions/956047/serialize-quaternion-or-vector3.html
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.Serialization;
using System;

namespace CustomAnimation
{

 
 /// <summary>
 /// Since unity doesn't flag the Vector3 as serializable, we
 /// need to create our own version. This one will automatically convert
 /// between Vector3 and SerializableVector3
 /// </summary>
 [System.Serializable]
 public struct SerializableVector3
 {
     /// <summary>
     /// x component
     /// </summary>
     public float x;
     
     /// <summary>
     /// y component
     /// </summary>
     public float y;
     
     /// <summary>
     /// z component
     /// </summary>
     public float z;
     
     /// <summary>
     /// Constructor
     /// </summary>
     /// <param name="rX"></param>
     /// <param name="rY"></param>
     /// <param name="rZ"></param>
     public SerializableVector3(float rX, float rY, float rZ)
     {
         x = rX;
         y = rY;
         z = rZ;
     }
     
     /// <summary>
     /// Returns a string representation of the object
     /// </summary>
     /// <returns></returns>
     public override string ToString()
     {
         return String.Format("[{0}, {1}, {2}]", x, y, z);
     }
     
     /// <summary>
     /// Automatic conversion from SerializableVector3 to Vector3
     /// </summary>
     /// <param name="rValue"></param>
     /// <returns></returns>
     public static implicit operator Vector3(SerializableVector3 rValue)
     {
         return new Vector3(rValue.x, rValue.y, rValue.z);
     }
     
     /// <summary>
     /// Automatic conversion from Vector3 to SerializableVector3
     /// </summary>
     /// <param name="rValue"></param>
     /// <returns></returns>
     public static implicit operator SerializableVector3(Vector3 rValue)
     {
         return new SerializableVector3(rValue.x, rValue.y, rValue.z);
     }
 }


 
 /// <summary>
 /// Since unity doesn't flag the Vector3 as serializable, we
 /// need to create our own version. This one will automatically convert
 /// between Vector3 and SerializableVector3
 /// </summary>
 [System.Serializable]
 public struct SerializableVector4
 {
     /// <summary>
     /// x component
     /// </summary>
     public float x;
     
     /// <summary>
     /// y component
     /// </summary>
     public float y;
     
     /// <summary>
     /// z component
     /// </summary>
     public float z;
     /// <summary>
     /// z component
     /// </summary>
     public float w;
     
     /// <summary>
     /// Constructor
     /// </summary>
     /// <param name="rX"></param>
     /// <param name="rY"></param>
     /// <param name="rZ"></param>
     public SerializableVector4(float rX, float rY, float rZ, float rW)
     {
         x = rX;
         y = rY;
         z = rZ;
         w = rW;
     }
     
     /// <summary>
     /// Returns a string representation of the object
     /// </summary>
     /// <returns></returns>
     public override string ToString()
     {
         return String.Format("[{0}, {1}, {2}]", x, y, z);
     }
     
     /// <summary>
     /// Automatic conversion from SerializableVector3 to Vector3
     /// </summary>
     /// <param name="rValue"></param>
     /// <returns></returns>
     public static implicit operator Vector4(SerializableVector4 rValue)
     {
         return new Vector4(rValue.x, rValue.y, rValue.z, rValue.w);
     }
     
     /// <summary>
     /// Automatic conversion from Vector3 to SerializableVector3
     /// </summary>
     /// <param name="rValue"></param>
     /// <returns></returns>
     public static implicit operator SerializableVector4(Vector4 rValue)
     {
         return new SerializableVector4(rValue.x, rValue.y, rValue.z, rValue.w);
     }
 }



 
 /// <summary>
 /// Since unity doesn't flag the Quaternion as serializable, we
 /// need to create our own version. This one will automatically convert
 /// between Quaternion and SerializableQuaternion
 /// </summary>
 [System.Serializable]
 public struct SerializableQuaternion
 {
     /// <summary>
     /// x component
     /// </summary>
     public float x;
     
     /// <summary>
     /// y component
     /// </summary>
     public float y;
     
     /// <summary>
     /// z component
     /// </summary>
     public float z;
     
     /// <summary>
     /// w component
     /// </summary>
     public float w;
     
     /// <summary>
     /// Constructor
     /// </summary>
     /// <param name="rX"></param>
     /// <param name="rY"></param>
     /// <param name="rZ"></param>
     /// <param name="rW"></param>
     public SerializableQuaternion(float rX, float rY, float rZ, float rW)
     {
         x = rX;
         y = rY;
         z = rZ;
         w = rW;
     }
     
     /// <summary>
     /// Returns a string representation of the object
     /// </summary>
     /// <returns></returns>
     public override string ToString()
     {
         return String.Format("[{0}, {1}, {2}, {3}]", x, y, z, w);
     }
     
     /// <summary>
     /// Automatic conversion from SerializableQuaternion to Quaternion
     /// </summary>
     /// <param name="rValue"></param>
     /// <returns></returns>
     public static implicit operator Quaternion(SerializableQuaternion rValue)
     {
         return new Quaternion(rValue.x, rValue.y, rValue.z, rValue.w);
     }
     
     /// <summary>
     /// Automatic conversion from Quaternion to SerializableQuaternion
     /// </summary>
     /// <param name="rValue"></param>
     /// <returns></returns>
     public static implicit operator SerializableQuaternion(Quaternion rValue)
     {
         return new SerializableQuaternion(rValue.x, rValue.y, rValue.z, rValue.w);
     }
 }



        [System.Serializable]
    public class StartPose
    {
        public Vector3 position;
        public Quaternion orientation;
        public bool forceWalkEndConstraints;
    }

    [System.Serializable]
    public class CKeyframe{
        
		public Vector4[] rotations;      
        public Vector3[] positions;
        public float[] ma;
        public float[] mf;
        public float[] mr;
        public float[] tl;
        public bool reset;
    }


    [System.Serializable]
    public class BKeyframe{
        
		public Vector4[] rotations;      
        public Vector3[] positions;
        public float[] ma;
        public float[] mf;
        public float[] mr;
        public float[] tl;
    }


     sealed class Vector3SerializationSurrogate : ISerializationSurrogate {
     
        // Method called to serialize a Vector3 object
        public void GetObjectData(System.Object obj,
                                SerializationInfo info, StreamingContext context) {
            
            Vector3 v3 = (Vector3) obj;
            info.AddValue("x", v3.x);
            info.AddValue("y", v3.y);
            info.AddValue("z", v3.z);
        }
        
        // Method called to deserialize a Vector3 object
        public System.Object SetObjectData(System.Object obj,
                                            SerializationInfo info, StreamingContext context,
                                            ISurrogateSelector selector) {
            
            Vector3 v3 = (Vector3) obj;
            v3.x = (float)info.GetValue("x", typeof(float));
            v3.y = (float)info.GetValue("y", typeof(float));
            v3.z = (float)info.GetValue("z", typeof(float));
            obj = v3;
            return obj;   // Formatters ignore this return value //Seems to have been fixed!
        }
    }

    

     sealed class Vector4SerializationSurrogate : ISerializationSurrogate {
     
        // Method called to serialize a Vector3 object
        public void GetObjectData(System.Object obj,
                                SerializationInfo info, StreamingContext context) {
            
            Vector4 v4 = (Vector4) obj;
            info.AddValue("x", v4.x);
            info.AddValue("y", v4.y);
            info.AddValue("z", v4.z);
            info.AddValue("w", v4.w);
        }
        
        // Method called to deserialize a Vector3 object
        public System.Object SetObjectData(System.Object obj,
                                            SerializationInfo info, StreamingContext context,
                                            ISurrogateSelector selector) {
            
            Vector4 v4 = (Vector4) obj;
            v4.x = (float)info.GetValue("x", typeof(float));
            v4.y = (float)info.GetValue("y", typeof(float));
            v4.z = (float)info.GetValue("z", typeof(float));
            v4.w = (float)info.GetValue("w", typeof(float));
            obj = v4;
            return obj;   // Formatters ignore this return value //Seems to have been fixed!
        }
    }

    
}
