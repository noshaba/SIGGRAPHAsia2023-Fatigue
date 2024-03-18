using System;
using System.IO;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.AI;
using System.Net.Sockets;
using System.Text;
using System.Threading;


namespace CustomAnimation
{   
   
    public class BufferedAnimationTCPClient : BufferedNetworkPoseProvider
    {
        private TcpClient socketConnection = null;
        private Thread clientReceiveThread;
        public string url = "localhost";
        public int port = 8888;
        public int bufferSize = 10485760;
        public float visScaleFactor;
        protected bool initialized;
        public List<string> messageQueue;
        Dictionary<string, GameObject> points;
        StartPose startPose;

        public bool paused;
        public bool switchUpAxis;
        public bool createVisBodies = true;
        public string filename = "";
        public void Start()
        {
            data = new BufferData();
            data.poseBuffer = new CKeyframe[data.frameBufferSize];
            points = new Dictionary<string, GameObject>();
            initialized = false;
            data.skeletonDesc = null;
            messageQueue = new List<string>();
            startPose = new StartPose();
            startPose.forceWalkEndConstraints = false;
            startPose.position = gameObject.transform.position;
            startPose.position.x = -startPose.position.x;
            
      
            var q = transform.rotation;
            // switch to MG coordinate system: unity is left-handed y-up, MG right-handed y-up
            // See: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
            q.Set(-q.x, q.y, q.z, -q.w);
            startPose.orientation = q;
            ConnectToTcpServer();
           
        }

        public void OnDestroy()
        {
            Debug.Log("Destroy object");
            if (socketConnection != null) socketConnection.Close();
        }


        /// <summary> 	
        /// Setup socket connection. 	
        /// </summary> 	
        private void ConnectToTcpServer()
        {
            try
            {
                socketConnection = new TcpClient(url, port);
                var positionStr = JsonUtility.ToJson(startPose);
                SendMessageToServer("SetPose:" + positionStr);
                clientReceiveThread = new Thread(new ThreadStart(ListenForData));
                clientReceiveThread.IsBackground = true;
                clientReceiveThread.Start();
            }
            catch (Exception e)
            {
                Debug.Log("On client connect exception " + e);
            }

        }

        /// <summary> 	
        /// Runs in background clientReceiveThread; Listens for incomming data. 	
        /// </summary>     
        private void ListenForData()
        {

            Debug.Log("Start listening for data");
            try
            {
                //Byte[] bytes = new Byte[bufferSize];
                byte[] buffer = new byte[bufferSize];
                while (true)
                {

                    // Get a stream object for reading 				
                    using (NetworkStream stream = socketConnection.GetStream())
                    {
                        int length;
                        // Read incomming stream into byte arrary. 					
                        while ((length = stream.Read(buffer, 0, buffer.Length)) != 0)
                        {
                            string serverMessage = "";
                            //Debug.Log("readSoFar " + length.ToString());
                            int startOffset = 0;
                            int endOffset = 0;
                            while (startOffset < buffer.Length && endOffset < buffer.Length)
                            {
                                while (buffer[endOffset] != 0x00 && endOffset < buffer.Length)
                                {

                                    endOffset++;
                                }
                                int incomDataLength = endOffset - startOffset;
                                //Debug.Log("process " + startOffset.ToString() + " " + endOffset.ToString() + " " + incomDataLength.ToString());
                                if (incomDataLength <= 0)
                                {
                                    break;
                                }
                                var incomingData = new byte[incomDataLength];

                                Array.Copy(buffer, startOffset, incomingData, 0, incomDataLength);
                                // Convert byte array to string message. 						
                                serverMessage += Encoding.ASCII.GetString(incomingData);


                                if (buffer[endOffset] == 0x00)
                                {
                                    startOffset = endOffset + 1;
                                    //incomDataLength =  offset - 1;
                                }
                            }
                            //Debug.Log("server message received as: " + serverMessage);
                            if (initialized && serverMessage != "")
                            {
                                setPoseFromString(serverMessage);
                            }
                            else if (data.skeletonDesc == null && serverMessage != "")
                            {
                                ProcessSkeletonString(serverMessage);
                            }

                            var message = "OK";
                            if (initialized && messageQueue.Count > 0)
                            {
                                message = messageQueue[0];
                                messageQueue.RemoveAt(0);

                            }
                            SendMessageToServer(message);
                        }
                    }


                }
            }
            catch (SocketException socketException)
            {
                Debug.Log("Socket exception: " + socketException);
            }
        }

        /// <summary> 	
        /// Send message to server using socket connection. 	
        /// </summary> 	
        public void SendMessageToServer(string clientMessage)
        {

            if (socketConnection == null)
            {
                return;
            }
            try
            {
                // Get a stream object for writing. 			
                NetworkStream stream = socketConnection.GetStream();
                if (stream.CanWrite)
                {

                    // Convert string message to byte array.                 
                    byte[] clientMessageAsByteArray = Encoding.ASCII.GetBytes(clientMessage + "\x0");

                    // Write byte array to socketConnection stream.                 
                    stream.Write(clientMessageAsByteArray, 0, clientMessageAsByteArray.Length);
                    //Debug.Log("Client sent his message - should be received by server");
                }
            }
            catch (SocketException socketException)
            {
                Debug.Log("Socket exception: " + socketException);
            }

        }

        private void setPoseFromString(string message)
        {
            if (paused) return; 
            CKeyframe pose = JsonUtility.FromJson<CKeyframe>(message);
            for(int i =0; i <pose.positions.Length; i++)
            {
        
                var p = pose.positions[i]* scaleFactor;
                var r = pose.rotations[i];
                if (switchUpAxis){
                    p = new Vector3(p.x, p.z, p.y);
                    r = new Vector4(r.x, r.z, r.y, -r.w);
                }
                pose.positions[i] = p;
                pose.rotations[i] = r;
            }
            AddFrame(pose);
            
        }


        public void Update()
        {

            if (paused) return;
            CKeyframe pose = null;
            if (data.head >=0)
                pose = data.poseBuffer[data.head];
            if (initialized && pose != null)
            {
                int index = 0;
                foreach (var name in data.skeletonDesc.jointSequence)
                {
                    
                    var r = pose.rotations[index];
                    points[name].transform.position =  pose.positions[index];
                    points[name].transform.rotation = new Quaternion(r.x, r.y, r.z, r.w);

                    index += 1;
                }
            }
            else if (!initialized && data.skeletonDesc != null)
            {
                initSkeleton();
            }


        }


        public void ProcessSkeletonString(string skeletonString)
        {
            Debug.LogWarning("process skeleton string" + skeletonString);
            try
            {
                Debug.LogWarning("try to construct skeleton desc");
                data.skeletonDesc = JsonUtility.FromJson<SkeletonDesc>(skeletonString);

                Debug.LogWarning("construct skeleton desc"+data.skeletonDesc.jointSequence.ToString());
                buildPoseParameterIndexMap(data.skeletonDesc.jointSequence);
                Debug.LogWarning("buildPoseParameterIndexMap");
            }
            catch (Exception e)
            {

                Debug.LogWarning("exception" + e.Message);
                return;
            }

        }
        public void initSkeleton()
        {
            if (data.skeletonDesc != null)
            {
                
                Debug.Log("generated from skeleton desc");
                createDebugSkeleton(data.skeletonDesc);
                
                initialized = true;
            }
            Debug.Log("processed skeleton");
        }

     
        public void createDebugSkeleton(SkeletonDesc skeletonDesc)
        {
            Debug.Log("createDebugSkeleton");
            foreach (var jointName in data.skeletonDesc.jointSequence)
            {

                if (createVisBodies){
                    points[jointName] = createJointVisualization(visScaleFactor);
                }else{
                    points[jointName] = new GameObject();
                }
                points[jointName].name = jointName;
                points[jointName].transform.parent = transform;
            }
        }

        protected GameObject createJointVisualization(float scale=1.0f)
        {
            GameObject jointObj = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            Mesh mesh = jointObj.GetComponent<MeshFilter>().mesh;
            List<Vector3> newVertices = new List<Vector3>();
            var r = Quaternion.Euler(90, 0, 0);
            for (int i = 0; i < mesh.vertices.Length; i++)
            {
                var v = mesh.vertices[i];
                var newV =  new Vector3(v.x * scale, scale * v.y, v.z * scale);
                newVertices.Add(newV);
            }
            jointObj.GetComponent<MeshFilter>().mesh.vertices = newVertices.ToArray();
            jointObj.GetComponent<MeshFilter>().mesh.RecalculateNormals();

            return jointObj;
        }

        protected void buildPoseParameterIndexMap(string[] jointSequence)
        {

            indexMap  = new Dictionary<string, int>();
            foreach (string name in jointSequence)
            {
                int idx = Array.IndexOf(jointSequence, name);
                indexMap[name] = idx;
            }
        }

        public void SaveAnimation()
        {
            var _filename = Path.Combine(Application.streamingAssetsPath, filename);
            base.SaveToBinaryFile(_filename);
        }
    }



}
