// import React, { useState } from 'react';
// import './InputBox.css';

// const InputBox = ({ sendMessage }) => {
//   const [message, setMessage] = useState('');

//   const handleSend = () => {
//     sendMessage(message);
//     setMessage('');
//   };

//   return (
//     <div className="input-box">
//       <input
//         type="text"
//         value={message}
//         onChange={(e) => setMessage(e.target.value)}
//         placeholder="Type a message..."
//       />
//       <button onClick={handleSend}>Send</button>
//     </div>
//   );
// };

// export default InputBox;

import React, { useState } from 'react';
// import './InputBox.css';
import './Bot.css';

const InputBox = ({ sendMessage }) => {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    if (message.trim()) {
      sendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div className="input-box">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type a message..."
        onKeyPress={handleKeyPress}
      />
      <button onClick={handleSend}>Send</button>
    </div>
  );
};

export default InputBox;
