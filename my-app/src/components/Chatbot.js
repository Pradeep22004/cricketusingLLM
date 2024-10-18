import React, { useState, useEffect } from 'react';
import "./Chatbot.css";
const Chatbot = () => {
  const [data, setData] = useState(null);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/data.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => setData(data))
      .catch(error => {
        console.error('Error fetching data:', error);
        setError(error.message);
      });
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (data) {
      const found = data.questions.find(q => q.question.toLowerCase().includes(input.toLowerCase()));
      if (found) {
        setMessages([...messages, { text: input, type: 'user' }, { text: found.answer, type: 'bot' }]);
      } else {
        setMessages([...messages, { text: input, type: 'user' }, { text: "Sorry, I don't have an answer for that.", type: 'bot' }]);
      }
      setInput('');
    }
  };

  if (error) return <div>Error: {error}</div>;
  if (!data) return <div>Loading...</div>;

  return (
    <div>
      <div className="chatbox">
        {messages.map((msg, index) => (
          <div key={index} className={msg.type}>
            {msg.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me a question..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default Chatbot;
