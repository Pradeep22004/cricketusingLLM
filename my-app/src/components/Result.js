import React from 'react';
import ResultExplained from './ResultExplained';

function Result(props) {

  function renderQuestions(key, index) {
    return (
      <ResultExplained
        key={key.question}
        questionContent={key.question}
        correctAns={key.correctAns}
        userAnswer={props.userAns[index]}
      />
    );
  }

  return (
    <div className="quizResult">
      <div className="resultHeader">
        <h3>Hey {props.username},</h3>
        <h2>You Scored {((100 * props.correct) / (props.correct + props.incorrect)).toFixed(2)}% Correct!</h2>
      </div>
      <div className="statistics">
        <span><h1>{props.correct}</h1><p>Correct</p></span>
        <span><h1>{props.incorrect}</h1><p>Incorrect</p></span>
      </div>
      <p className="reviewline">Review of your answers:</p>
      <ul className="resultQuestions">
        {props.questionData.map(renderQuestions)}
      </ul>
      <div className="attemptdiv">
        <button onClick={props.onRestart} className="attempt">Attempt Again</button>
      </div>
    </div>
  );
}

export default Result;
