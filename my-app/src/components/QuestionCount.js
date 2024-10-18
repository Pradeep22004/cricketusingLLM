import React from 'react';

function QuestionCount(props) {
  return (
    <div className="questionCount">
      <p>#<span>{props.counter}</span> / <span>{props.total}</span></p>
    </div>
  );

}

export default QuestionCount;
