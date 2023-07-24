import React from 'react';

const VideoPlayer = ({ videoUrl }) => {
  return (
    <div >
      <h2>Video Player</h2>
      <video controls width="640" height="360">
        <source src={videoUrl} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default VideoPlayer;