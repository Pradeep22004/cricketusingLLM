import React from 'react';

function ResultExplained(props) {
  var ansStatus = 'correct';
  if(props.correctAns !== props.userAnswer){
    ansStatus = 'incorrect';
  }
  function outputAns(){
    if(props.correctAns !== props.userAnswer){
      return(
        "Your Answer: "+props.userAnswer
      );
    }
  }

  return (
    <li className={ansStatus}>
      <b>{props.questionContent}</b>
      <div>
      Correct Answer: {props.correctAns}<br/>
      {outputAns()}
      </div>
    </li>
  );

}

export default ResultExplained;
