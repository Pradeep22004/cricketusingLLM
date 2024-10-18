import React from 'react';

function AnswerOption(props) {

  return (
    <li className="answerOption">
      <input
        type="radio"
        id={props.answerContent}
        value={props.answerContent}
        disabled={props.answer}
        onChange={props.onAnswerSelected}
        className="option-input radio"
      />
      <label htmlFor={props.answerContent}>
        {props.answerContent}
      </label>
    </li>
  );

}

export default AnswerOption;
