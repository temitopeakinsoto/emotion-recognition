import React, { useRef, useEffect } from 'react';

const WebcamVideo = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    let mediaStream = null;
    const constraints = { video: true };

    const getUserMedia = async () => {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };

    getUserMedia();

    return () => {
      // Stop the video stream when the component unmounts
      if (mediaStream) {
        const tracks = mediaStream.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  const handleStopStream = () => {
    // Stop the video stream when the stop button is clicked
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
    }
  };

  return (
    <div>
      <h2>Webcam Video</h2>
      <video ref={videoRef} autoPlay playsInline />
      <button onClick={handleStopStream}>Stop Stream</button>
    </div>
  );
};

export default WebcamVideo;
