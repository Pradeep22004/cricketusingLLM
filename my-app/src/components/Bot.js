// import React, { useState } from 'react';
// import Header from './Header1';
// import ChatWindow from './ChatWindow';
// import InputBox from './InputBox';
// import './Bot.css';
// import axios from 'axios';
// const Bot = () => {
//   const [messages, setMessages] = useState([]);
//   const [question, setQuestion] = useState('');
//   const [response, setResponse] = useState('');

//   const sendMessage = (message) => {
//     setMessages([...messages, { sender: 'user', text: message }]);
//     // Simulate LLM response
//     setTimeout(() => {
//       setMessages((prevMessages) => [
//         ...prevMessages,
//         { sender: 'bot', text: `Response to: ${message}` },
//       ]);
//     }, 1000);
//   };
//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       const res = await axios.post('http://localhost:3001/ask', { question });
//       setResponse(res.data.response);
//     } catch (error) {
//       console.error("Error fetching the response", error);
//       setResponse("Error fetching the response");
//     }
//   };


//   return (
//     <div className="app">
//       <Header />
//       <ChatWindow messages={messages} />
//       <InputBox sendMessage={sendMessage} />
//       <div className="response">
//           <h2>Response:</h2>
//           <p>{response}</p>
//         </div>

//     </div>
//   );
// };

// export default Bot;
// import React, { useState } from 'react';
// import Header from './Header1';
// import ChatWindow from './ChatWindow';
// import InputBox from './InputBox';
// import './Bot.css';
// import axios from 'axios';

// const Bot = () => {
//   const [messages, setMessages] = useState([]);

//   const sendMessage = async (message) => {
//     setMessages((prevMessages) => [...prevMessages, { sender: 'user', text: message }]);
//     try {
//       const res = await axios.post('http://localhost:5000/ask', { question: message });
//       setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: res.data.response }]);
//     } catch (error) {
//       console.error("Error fetching the response", error);
//       setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: "Error fetching the response" }]);
//     }
//   };

//   return (
//     <div className="app">
//       <Header />
//       <ChatWindow messages={messages} />
//       <InputBox sendMessage={sendMessage} />
//     </div>
//   );
// };

// export default Bot;


import React, { useState } from 'react';
import Header from './Header1';
import ChatWindow from './ChatWindow';
import InputBox from './InputBox';
import './Bot.css';
import axios from 'axios';

const Bot = () => {
  const [messages, setMessages] = useState([]);

  const sendMessage = async (message) => {
    setMessages((prevMessages) => [...prevMessages, { sender: 'user', text: message }]);
    try {
      const res = await axios.post('http://localhost:5000/ask', { question: message });
      setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: res.data.response }]);
    } catch (error) {
      console.error("Error fetching the response", error);
      setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: "Error fetching the response" }]);
    }
  };

  return (
    <div className="app">
      <Header className="header" />
      <ChatWindow className="chat-window" messages={messages} />
      <InputBox className="input-box" sendMessage={sendMessage} />
    </div>
  );
};

export default Bot;
