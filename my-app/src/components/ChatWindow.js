// import React from 'react';
// import Message from './Message';
// import './ChatWindow.css';

// const ChatWindow = ({ messages }) => {
//   return (
//     <div className="chat-window">
//       {messages.map((msg, index) => (
//         <Message key={index} sender={msg.sender} text={msg.text} />
//       ))}
//     </div>
//   );
// };

// export default ChatWindow;
import React from 'react';
import './ChatWindow.css';

const ChatWindow = ({ messages }) => {
  return (
    <div className="chat-window">
      {messages.map((msg, index) => (
        <div key={index} className={`message ${msg.sender}`}>
          {/* <span>{msg.text}</span> */}
          {msg.text}
        </div>
      ))}
    </div>
  );
};

export default ChatWindow;

