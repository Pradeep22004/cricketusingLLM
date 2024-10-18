import React from 'react';
import imag from './../quiz.png';

function Start(props) {

  return (
    <div className="Start">
      <div className="heading">
       <img src={imag} alt="" /><h2>Cricket Quiz</h2>
      </div>
      <form onSubmit={props.usernameSubmitted} className="username">
        <label>
          <input type="text" value={props.username}  onChange={props.usernameValue} placeholder="Enter Your Name..." required/>
        </label>
        <input type="submit" value="PLAY" />
      </form>
    </div>

  );

}


export default Start;
