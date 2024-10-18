import React from 'react';
import PlayerCard from './PlayerCard';
// import './App.css';
import './PlayerCard.css'
const players = [
  {
    name: 'Rohit Sharma',
    photo: 'rohit.png',
    battingStyle: 'Right-handed',
    age: 36,
    contribution: 'Batter',
  },
  {
    name:'Virat Kohli',
    photo: 'kohli.png',
    battingStyle: 'Right-handed',
    age: 35,
    contribution: 'Batter',
  },
  {
    name:'Jasprit Bumrah',
    photo: 'bumrah.png',
    battingStyle: 'Right-handed',
    age: 30,
    contribution: 'Bowler',
  },
  {
    name:'Ravindra Jadeja',
    photo: 'jadeja.png',
    battingStyle: 'Left-handed',
    age: 30,
    contribution: 'Batter',
  },
  {
    name:'Ravichandran Ashwin',
    photo: 'ashwin.png',
    battingStyle: 'Right-handed',
    age: 37,
    contribution: 'Bowler(Spin)',
  },
  {
    name:'Mohammad Shami',
    photo: 'shami.png',
    battingStyle: 'Right-handed',
    age: 33,
    contribution: 'Bowler',
  },
  {
    name:'Mohammed Siraj',
    photo: 'siraj.png',
    battingStyle: 'Right-handed',
    age: 30,
    contribution: 'Bowler',
  },
  {
    name:'KL Rahul',
    photo: 'rahul.png',
    battingStyle: 'Right-handed',
    age: 31,
    contribution: 'Batter(WK)',
  },
  {
    name:'Shubman Gill',
    photo: 'gill.png',
    battingStyle: 'Right-handed',
    age: 24,
    contribution: 'Batter',
  },
  {
    name:'Hardik Pandya',
    photo: 'pandya.png',
    battingStyle: 'Right-handed',
    age: 30,
    contribution: 'All Rounder',
  },
  // Add more player objects here
];

const card = () => {
  const appStyles={
    fontFamily: 'Arial, sans-serif',
    backgroundcolor:'',
    padding: '20px',
  };

  return (
    <div className="App" style={appStyles}>
      <h1>Cricketers Profiles</h1>
      <div className="player-cards">
        {players.map((player, index) => (
          <PlayerCard key={index} player={player} />
        ))}
      </div>
    </div>
  );
};

export default card;