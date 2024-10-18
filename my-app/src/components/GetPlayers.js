// src/App.js
import React, { useState } from 'react';
import axios from 'axios';

function GetPlayers() {
  const [teamId, setTeamId] = useState('');
  const [players, setPlayers] = useState([]);

  const fetchPlayers = async (teamId) => {
    try {
      const response = await axios.get(`http://localhost:3000/api/players/${teamId}`);
      setPlayers(response.data);
    } catch (error) {
      console.error('Error fetching players:', error);
    }
  };

  const handleTeamClick = (id) => {
    setTeamId(id);
    fetchPlayers(id);
  };

  return (
    <div>
    <h1>Cricket Teams</h1>
    <button onClick={() => handleTeamClick('15826')}>Team 1</button>
    {/* Add more buttons for different teams */}
    
    <h2>Players</h2>
    <ul>
      {players.map((player) => (
        <li key={player.id}>
          <img src={`https://www.cricbuzz.com/images/players/${player.imageId}.jpg`} alt={player.name} onError={(e) => { e.target.onerror = null; e.target.src = 'https://via.placeholder.com/150'; }} />
          <p>Name: {player.name}</p>
          <p>Role: {player.role}</p>
          <p>Batting Style: {player.battingStyle}</p>
          <p>Bowling Style: {player.bowlingStyle}</p>
        </li>
      ))}
    </ul>
  </div>
  );
}

export default GetPlayers;
