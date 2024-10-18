

import React, { useState } from 'react';
import axios from 'axios';
// import { Center } from '@chakra-ui/react';

const PlayerSearch = () => {
  const [playerName, setPlayerName] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [error, setError] = useState('');

  const searchPlayers = async () => {
    try {
      const response = await axios.get(`http://localhost:3000/api/search/${playerName}`);
      const players = response.data.player;
      setSearchResults(players);
      setError('');
    } catch (err) {
      console.error('Error searching players:', err);
      // setError('Failed to search for players');

      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else {
        setError('Failed to search for players');
      }

    }
  };

  const fetchPlayerDetails = async (playerId) => {
    try {
      const response = await axios.get(`http://localhost:3000/api/player/${playerId}`);
      setSelectedPlayer(response.data);
      setError('');
    } catch (err) {
      console.error('Error fetching player details:', err);
      // setError('Failed to fetch player details');

      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } 
      else {
        setError('Failed to fetch player details');
      }
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.searchBox}>
        <input
          type="text"
          value={playerName}
          onChange={(e) => setPlayerName(e.target.value)}
          placeholder="Enter Player Name"
          style={styles.input}
        />
        <button onClick={searchPlayers} style={styles.button}>Search Players</button>
      </div>

      {error && <p style={styles.error}>{error}</p>}

      {searchResults.length > 0 && (
        <div style={styles.resultsContainer}>
          <h2 style={styles.heading}>Search Results:</h2>
          <ul style={styles.list}>
            {searchResults.map((player) => (
              <li key={player.id} style={styles.listItem}>
                <img src={`http://i.cricketcb.com/stats/img/faceImages/${player.faceImageId}.jpg`} alt={player.name} style={styles.playerImage} />
                <p>{player.name} ({player.teamName})</p>
                <button onClick={() => fetchPlayerDetails(player.id)} style={styles.button}>Get Player Info</button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {selectedPlayer && (
        <div style={styles.playerDetails}>
          <h1>{selectedPlayer.name} ({selectedPlayer.nickName})</h1>
          <div style={styles.centeredImageContainer}>
            <img src={selectedPlayer.image} alt={selectedPlayer.name} style={styles.centeredImage} />
          </div>
          <p><strong>Role:</strong> {selectedPlayer.role}</p>
          <p><strong>Batting Style:</strong> {selectedPlayer.bat}</p>
          <p><strong>Bowling Style:</strong> {selectedPlayer.bowl}</p>
          <p><strong>Height:</strong> {selectedPlayer.height}</p>
          <p><strong>Birth Place:</strong> {selectedPlayer.birthPlace}</p>
          <p><strong>Date of Birth:</strong> {selectedPlayer.DoB}</p>
          <p><strong>International Team:</strong> {selectedPlayer.intlTeam}</p>
          <p><strong>Teams:</strong> {selectedPlayer.teams}</p>
          <h3>Rankings</h3>
          <p><strong>Test Rank:</strong> {selectedPlayer.rankings.bat.testRank}</p>
          <p><strong>Best Test Rank:</strong> {selectedPlayer.rankings.bat.testBestRank}</p>
          <p><strong>Best ODI Rank:</strong> {selectedPlayer.rankings.bat.odiBestRank}</p>
          <p><strong>Best T20 Rank:</strong> {selectedPlayer.rankings.bat.t20BestRank}</p>
          <h3>Bio</h3>
          <p dangerouslySetInnerHTML={{ __html: selectedPlayer.bio }}></p>
        </div>
      )}
    </div>
  );
};

const styles = {
    h1:{
        fontSize:'40px',
    },
  container: {
    fontFamily: 'Arial, sans-serif',
    padding: '20px',
    maxWidth: '800px',
    margin: '0 auto',
    backgroundColor: '#c7f3ff',
    borderRadius: '8px',
    boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
  },
  searchBox: {
    display: 'flex',
    justifyContent: 'center',
    marginBottom: '20px',
  },
  input: {
    padding: '10px',
    fontSize: '16px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    marginRight: '10px',
    width: 'calc(100% - 110px)', // Adjust width to fit button next to it
  },
  button: {
    padding: '10px 20px',
    fontSize: '16px',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: '#3fc5f0',
    color: '#fff',
    cursor: 'pointer',
    transition: 'background-color 0.3s ease',
  },
  buttonHover: {
    backgroundColor: '#0056b3',
  },
  error: {
    color: 'red',
    textAlign: 'center',
  },
  resultsContainer: {
    marginTop: '20px',
  },
  heading: {
    fontSize: '24px',
    textAlign: 'center',
    marginBottom: '20px',
  },
  list: {
    listStyleType: 'none',
    padding: '0',
  },
  listItem: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '10px',
    padding: '10px',
    backgroundColor: '#fff',
    borderRadius: '4px',
    boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
  },
  playerImage: {
    marginRight: '10px',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    objectFit: 'cover',
  },
  playerImage1:{
    borderRadius: '50%',
    
    
  },
  centeredImageContainer: {
    display: 'flex',
    justifyContent: 'center',
    marginBottom: '20px',
  },
  centeredImage: {
    width: '150px',
    height: '150px',
    borderRadius: '50%',
    objectFit: 'cover',
  },
  playerDetails: {
    marginTop: '20px',
    padding: '20px',
    backgroundColor: '#fff',
    borderRadius: '4px',
    boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
  },
};

export default PlayerSearch;
