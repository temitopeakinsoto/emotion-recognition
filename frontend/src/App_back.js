import React, { useState, useEffect } from 'react';
import './App.css';
const url = 'http://localhost:3000/emotion'

const App = () => {
  const [emotionData, setEmotionData] = useState([]);
  const [stream, setStream] = useState(null); 


  useEffect(() => {
    // Start capturing video frames and sending them to the server
    const captureFramesAndSend = async () => {
      const videoElement = document.getElementById('videoStream');
      const canvasElement = document.createElement('canvas');
      const canvasContext = canvasElement.getContext('2d');

      try {
        const constraints = { video: true };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;

        const sendFramesToServer = async () => {
          // Capture a frame from the video stream and draw it on the canvas
          canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
          const frameBlob = await new Promise(resolve => canvasElement.toBlob(resolve, 'image/jpeg'));

          // Send the frame as a blob to the server
          const formData = new FormData();
          formData.append('frames', frameBlob);

          const response = await fetch(url, {
            method: 'POST',
            body: formData,
          });

          const data = await response.json();
          setEmotionData(data);

          // Schedule the next frame capture and send
          requestAnimationFrame(sendFramesToServer);
        };

        // Start sending frames to the server
        sendFramesToServer();
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };

    captureFramesAndSend();
  }, [stream]);

  const handleStopStream = () => {
    // Stop the video stream when the stop button is clicked
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      setStream(null);
    }
  };

  return (
    <div className="App">
      <h1>Emotion Analysis</h1>
      <div className="videoContainer">
        <video id="videoStream" width="640" height="480" autoPlay playsInline />
      </div>
      <div className="emotionResults">
        <h2>Emotion Results</h2>
        <ul>
          {emotionData.map((data, index) => (
            <li key={index}>
              Frame {index + 1}: {data.emotion} (Probability: {data.probability.toFixed(2)})
            </li>
          ))}
        </ul>
      </div>
      <button onClick={handleStopStream}>Stop Streaming</button>

    </div>
  );
};

export default App;
