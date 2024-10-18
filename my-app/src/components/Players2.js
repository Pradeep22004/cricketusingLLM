import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './Players2.css';

const Players = () => {
  const [players, setPlayers] = useState([]);

  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        const response = await axios.get('/api/players');
        setPlayers(response.data.player);
      } catch (error) {
        console.error('Error fetching players data:', error);
      }
    };

    fetchPlayers();
  }, []);

  return (
    <div className="container">
      <h1 className="heading">Cricket Players</h1>
      <table className="table">
        <thead>
          <tr>
            <th className="th">Name</th>
            <th className="th">Role</th>
            <th className="th">Batting Style</th>
            <th className="th">Bowling Style</th>
          </tr>
        </thead>
        <tbody>
          {players.map((player, index) => (
            player.isHeader ? (
              <tr key={index}>
                <td className="td" colSpan="4"><strong>{player.name}</strong></td>
              </tr>
            ) : (
              <tr className="tr" key={player.id}>
                <td className="td">{player.name}</td>
                <td className="td">{player.role}</td>
                <td className="td">{player.battingStyle}</td>
                <td className="td">{player.bowlingStyle}</td>
              </tr>
            )
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Players;
