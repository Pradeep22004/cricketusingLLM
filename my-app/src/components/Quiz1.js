import React, { Component } from 'react';
import quizQuestions from './QuizData';
import Quiz from './Quiz';
import Result from './Result';
import Start from './Start';
import '../App.css';
import './quiz.css';
class Quiz1 extends Component {

  constructor(props) {
    super(props);

    this.state = {
      username: '',
      startState: 0,
      counter: 0,
      questionId: 1,
      question: '',
      answerOptions: [],
      answer: '',
      correct: 0,
      incorrect: 0,
      userAnswer: [],
      result: 0,
      correctAns: '',
      selectedQuestions: [] // Add state for selected questions
    };

    this.handleAnswerSelected = this.handleAnswerSelected.bind(this);
    this.handleUsername = this.handleUsername.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleRestart = this.handleRestart.bind(this);
  }

  componentDidMount() {
    this.initializeQuiz();
  }

  initializeQuiz() {
    const shuffledQuestions = this.shuffleArray(quizQuestions);
    const selectedQuestions = shuffledQuestions.slice(0, 5);

    this.setState({
      selectedQuestions: selectedQuestions,
      question: selectedQuestions[0].question,
      answerOptions: selectedQuestions[0].answers,
      correctAns: selectedQuestions[0].correctAns,
      counter: 0,
      questionId: 1,
      answer: '',
      correct: 0,
      incorrect: 0,
      userAnswer: [],
      result: 0,
    });
  }

  shuffleArray(array) {
    let shuffledArray = array.slice();
    for (let i = shuffledArray.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffledArray[i], shuffledArray[j]] = [shuffledArray[j], shuffledArray[i]];
    }
    return shuffledArray;
  }

  handleSubmit(event) {
    event.preventDefault();
    this.setState({
      startState: 1
    });
  }

  handleUsername(event) {
    this.setState({
      username: event.currentTarget.value
    });
  }

  handleAnswerSelected(event) {
    const selectedAnswer = event.currentTarget.value;
    const isCorrect = selectedAnswer === this.state.correctAns;
    this.setState(prevState => ({
      correct: isCorrect ? prevState.correct + 1 : prevState.correct,
      incorrect: isCorrect ? prevState.incorrect : prevState.incorrect + 1,
      userAnswer: [...prevState.userAnswer, selectedAnswer],
    }), () => {
      if (this.state.questionId < this.state.selectedQuestions.length) {
        setTimeout(() => this.setNextQuestion(), 300);
      } else {
        setTimeout(() => this.setResults(), 300);
      }
    });
  }

  setNextQuestion() {
    const counter = this.state.counter + 1;
    const questionId = this.state.questionId + 1;

    this.setState({
      counter: counter,
      questionId: questionId,
      question: this.state.selectedQuestions[counter].question,
      answerOptions: this.state.selectedQuestions[counter].answers,
      correctAns: this.state.selectedQuestions[counter].correctAns,
      answer: ''
    });
  }

  setResults() {
    this.setState({ result: 1 });
  }

  handleRestart() {
    this.setState({
      username: '',
      startState: 0,
      counter: 0,
      questionId: 1,
      question: '',
      answerOptions: [],
      answer: '',
      correct: 0,
      incorrect: 0,
      userAnswer: [],
      result: 0,
      correctAns: '',
      selectedQuestions: []
    }, () => {
      this.initializeQuiz();
    });
  }

  renderQuiz() {
    if (!this.state.startState) {
      return (
        <Start username={this.state.username}
          usernameSubmitted={this.handleSubmit}
          usernameValue={this.handleUsername} />
      );
    } else {
      return (
        <Quiz
          answer={this.state.answer}
          answerOptions={this.state.answerOptions}
          questionId={this.state.questionId}
          question={this.state.question}
          questionTotal={this.state.selectedQuestions.length}
          onAnswerSelected={this.handleAnswerSelected}
        />
      );
    }
  }

  renderResult() {
    return (
      <Result username={this.state.username}
        correct={this.state.correct}
        incorrect={this.state.incorrect}
        questionData={this.state.selectedQuestions}
        userAns={this.state.userAnswer}
        onRestart={this.handleRestart} />
    );
  }

  render() {
    return (
      <div className="App">
        <div className="container">
          {this.state.result ? this.renderResult() : this.renderQuiz()}
        </div>
      </div>
    );
  }

}

export default Quiz1;
