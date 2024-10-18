
// import React from 'react';
// import { BrowserRouter as Router,Switch, Route,Routes } from 'react-router-dom';
// import './App.css';

// import Login from './components/Login';
// import Register from './components/Register';
// import Home1 from './components/Home1';
// import PlayerSearch from './components/PlayerSearch';
// import Card from './components/card'
// import PlayerCard from './components/PlayerCard';
// import Sidebar from './components/Sidebar';
// import Header from './components/Header';
// import Info from './components/PlayerInfo';
// import News from './components/News';
// import StatRecords from './components/StatsRecords';
// import Navbar from './components/Navbar';
// import Schedule from './components/Schedules';
// import Players from './components/Players2';
// import getPlayers from './components/GetPlayers';

// import Quiz from './components/Quiz1';
// import Career from './components/career';
// import Bot from './components/Bot';
// // import { News } from "./components/news/news";
// // import { ChakraProvider, Switch } from '@chakra-ui/react';
// function App() {
//   return (
    

    
//     <>
//     <Routes>

//         <Route path="/" element={<Login />}/>
//         <Route path="/register" element={<Register />} />
//     </Routes>
//       <Navbar/>
//       <Routes>
//         <Route path="/home1" element={<Home1 />}/>
//         <Route path="/getNews" element={<News />} />
//         <Route path="/playerInfo" element={<Info />} />
//         <Route path="/playerSearch" element={<PlayerSearch/>} />
//         <Route path="/records" element={<StatRecords/>} />
//         <Route path="/schedule" element={<Schedule/>} />
//         <Route path="/getPlayers" element={<Players/>} />
//         <Route path="/teamPlayers" element={<getPlayers/>} />
//         <Route path="/quiz" element={<Quiz/>} />
//         <Route path="/career" element={<Career/>} />
//         <Route path="/card" element={<Card />} />
//         <Route path="/cric-bot" element={<Bot />} />
//       </Routes>
//     </>
    


//   );
// }

// export default App;
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import './App.css';

import Login from './components/Login';
import Register from './components/Register';
import Home1 from './components/Home1';
import PlayerSearch from './components/PlayerSearch';
import Card from './components/card';
import PlayerCard from './components/PlayerCard';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Info from './components/PlayerInfo';
import News from './components/News';
import StatRecords from './components/StatsRecords';
import Navbar from './components/Navbar';
import Schedule from './components/Schedules';
import Players from './components/Players2';
import getPlayers from './components/GetPlayers';
import Quiz from './components/Quiz1';
import Career from './components/career';
import Bot from './components/Bot';
import Chatbot from './components/Chatbot';

function App() {
  const location = useLocation();
  const isAuthPage = location.pathname === '/' || location.pathname === '/register';

  return (
    <>
      {!isAuthPage && <Navbar />}
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/home1" element={<Home1 />} />
        <Route path="/getNews" element={<News />} />
        <Route path="/playerInfo" element={<Info />} />
        <Route path="/playerSearch" element={<PlayerSearch />} />
        <Route path="/records" element={<StatRecords />} />
        <Route path="/schedule" element={<Schedule />} />
        <Route path="/getPlayers" element={<Players />} />
        <Route path="/teamPlayers" element={<getPlayers />} />
        <Route path="/quiz" element={<Quiz />} />
        <Route path="/career" element={<Career />} />
        <Route path="/card" element={<Card />} />
        <Route path="/cric-bot" element={<Bot />} />
        <Route path="/bot" element={<Chatbot />} />
      </Routes>
    </>
  );
}

export default App;
