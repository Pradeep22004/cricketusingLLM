import React, { useState } from 'react';
import axios from 'axios';

const PlayerInfo = () => {
  const [playerId, setPlayerId] = useState('');
  const [playerData, setPlayerData] = useState(null);
  const [error, setError] = useState('');

  const fetchPlayerData = async () => {
    try {
      const response = await axios.get(`http://localhost:3000/api/player/${playerId}`);
      setPlayerData(response.data);
      setError('');
    } catch (err) {
      setError('Failed to fetch player data');
      setPlayerData(null);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={playerId}
        onChange={(e) => setPlayerId(e.target.value)}
        placeholder="Enter Player ID"
      />
      <button onClick={fetchPlayerData}>Get Player Info</button>

      {error && <p>{error}</p>}
      {playerData && (
        <div>
          <h2>{playerData.name} ({playerData.nickName})</h2>
          <img src={playerData.image} alt={playerData.name} />
          <p><strong>Role:</strong> {playerData.role}</p>
          <p><strong>Batting Style:</strong> {playerData.bat}</p>
          <p><strong>Bowling Style:</strong> {playerData.bowl}</p>
          <p><strong>Height:</strong> {playerData.height}</p>
          <p><strong>Birth Place:</strong> {playerData.birthPlace}</p>
          <p><strong>Date of Birth:</strong> {playerData.DoB}</p>
          <p><strong>International Team:</strong> {playerData.intlTeam}</p>
          <p><strong>Teams:</strong> {playerData.teams}</p>
          <h3>Rankings</h3>
          <p><strong>Test Rank:</strong> {playerData.rankings.bat.testRank}</p>
          <p><strong>Best Test Rank:</strong> {playerData.rankings.bat.testBestRank}</p>
          <p><strong>Best ODI Rank:</strong> {playerData.rankings.bat.odiBestRank}</p>
          <p><strong>Best T20 Rank:</strong> {playerData.rankings.bat.t20BestRank}</p>
          <h3>Bio</h3>
          <p dangerouslySetInnerHTML={{ __html: playerData.bio }}></p>
        </div>
      )}
    </div>
  );
};

export default PlayerInfo;


// import React, { useState } from 'react';
// import axios from 'axios';

// const PlayerInfo = () => {
//   const [playerName, setPlayerName] = useState('');
//   const [playerData, setPlayerData] = useState(null);
//   const [error, setError] = useState('');

//   const fetchPlayerData = async () => {
//     try {
//       const searchResponse = await axios.get(`http://localhost:3000/api/search/${playerName}`);
//       const players = searchResponse.data;
//       if (players.length === 0) {
//         setError('No players found');
//         setPlayerData(null);
//         return;
//       }
      
//       const playerId = players[0].id; // Assuming the first result is the correct player
//       const playerResponse = await axios.get(`http://localhost:3000/api/player/${playerId}`);
//       setPlayerData(playerResponse.data);
//       setError('');
//     } catch (err) {
//       setError('Failed to fetch player data');
//       setPlayerData(null);
//     }
//   };

//   return (
//     <div>
//       <input
//         type="text"
//         value={playerName}
//         onChange={(e) => setPlayerName(e.target.value)}
//         placeholder="Enter Player Name"
//       />
//       <button onClick={fetchPlayerData}>Get Player Info</button>

//       {error && <p>{error}</p>}
//       {playerData && (
//         <div>
//           <h2>{playerData.name} ({playerData.nickName})</h2>
//           <img src={playerData.image} alt={playerData.name} />
//           <p><strong>Role:</strong> {playerData.role}</p>
//           <p><strong>Batting Style:</strong> {playerData.bat}</p>
//           <p><strong>Bowling Style:</strong> {playerData.bowl}</p>
//           <p><strong>Height:</strong> {playerData.height}</p>
//           <p><strong>Birth Place:</strong> {playerData.birthPlace}</p>
//           <p><strong>Date of Birth:</strong> {playerData.DoB}</p>
//           <p><strong>International Team:</strong> {playerData.intlTeam}</p>
//           <p><strong>Teams:</strong> {playerData.teams}</p>
//           <h3>Rankings</h3>
//           <p><strong>Test Rank:</strong> {playerData.rankings.bat.testRank}</p>
//           <p><strong>Best Test Rank:</strong> {playerData.rankings.bat.testBestRank}</p>
//           <p><strong>Best ODI Rank:</strong> {playerData.rankings.bat.odiBestRank}</p>
//           <p><strong>Best T20 Rank:</strong> {playerData.rankings.bat.t20BestRank}</p>
//           <h3>Bio</h3>
//           <p dangerouslySetInnerHTML={{ __html: playerData.bio }}></p>
//         </div>
//       )}
//     </div>
//   );
// };

// export default PlayerInfo;
