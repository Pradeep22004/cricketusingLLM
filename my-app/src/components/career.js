// import React, { useEffect, useState } from 'react';
// import axios from 'axios';
// import './career.css';

// const App = () => {
//   const [trendingPlayers, setTrendingPlayers] = useState([]);
//   const [selectedPlayerId, setSelectedPlayerId] = useState(null);
//   const [playerCareer, setPlayerCareer] = useState([]);
//   const [playerBowling, setPlayerBowling] = useState({ headers: [], values: [] });
//   const [playerBatting, setPlayerBatting] = useState({ headers: [], values: [] });

//   useEffect(() => {
//     const fetchTrendingPlayers = async () => {
//       try {
//         const response = await axios.get('http://localhost:3001/api/trending-players');
//         setTrendingPlayers(response.data);
//       } catch (error) {
//         console.error('Error fetching trending players:', error);
//       }
//     };

//     fetchTrendingPlayers();
//   }, []);

//   useEffect(() => {
//     if (selectedPlayerId) {
//       const fetchPlayerData = async () => {
//         try {
//           const careerResponse = await axios.get(`http://localhost:3001/api/player/${selectedPlayerId}/career`);
//           const bowlingResponse = await axios.get(`http://localhost:3001/api/player/${selectedPlayerId}/bowling`);
//           const battingResponse = await axios.get(`http://localhost:3001/api/player/${selectedPlayerId}/batting`);

//           setPlayerCareer(careerResponse.data);
//           setPlayerBowling(bowlingResponse.data);
//           setPlayerBatting(battingResponse.data);
//         } catch (error) {
//           console.error('Error fetching player data:', error);
//         }
//       };

//       fetchPlayerData();
//     }
//   }, [selectedPlayerId]);

//   return (
//     <div className="App">
//       <h1>Trending Cricket Players</h1>
//       <select onChange={(e) => setSelectedPlayerId(e.target.value)}>
//         <option value="">Select a player</option>
//         {trendingPlayers.map((player) => (
//           <option key={player.id} value={player.id}>
//             {player.name} - {player.teamName}
//           </option>
//         ))}
//       </select>

//       {selectedPlayerId && (
//         <div className="player-stats">
//           <h2>Career</h2>
//           <ul>
//             {playerCareer.map((stat, index) => (
//               <li key={index}>
//                 {stat.name}: {stat.debut} - {stat.lastPlayed}
//               </li>
//             ))}
//           </ul>

//           <h2>Bowling</h2>
//           <table>
//             <thead>
//               <tr>
//                 {playerBowling.headers && playerBowling.headers.map((header, index) => (
//                   <th key={index}>{header}</th>
//                 ))}
//               </tr>
//             </thead>
//             <tbody>
//               {playerBowling.values && Array.isArray(playerBowling.values) && playerBowling.values.map((row, index) => (
//                 <tr key={index}>
//                   {row.values.map((value, i) => (
//                     <td key={i}>{value}</td>
//                   ))}
//                 </tr>
//               ))}
//             </tbody>
//           </table>

//           <h2>Batting</h2>
//           <table>
//             <thead>
//               <tr>
//                 {playerBatting.headers && playerBatting.headers.map((header, index) => (
//                   <th key={index}>{header}</th>
//                 ))}
//               </tr>
//             </thead>
//             <tbody>
//               {playerBatting.values && Array.isArray(playerBatting.values) && playerBatting.values.map((row, index) => (
//                 <tr key={index}>
//                   {row.values.map((value, i) => (
//                     <td key={i}>{value}</td>
//                   ))}
//                 </tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       )}
//     </div>
//   );
// };

// export default App;
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './career.css';

const App = () => {
  const [trendingPlayers, setTrendingPlayers] = useState([]);
  const [selectedPlayerId, setSelectedPlayerId] = useState(null);
  const [playerCareer, setPlayerCareer] = useState([]);
  const [playerBowling, setPlayerBowling] = useState({ headers: [], values: [] });
  const [playerBatting, setPlayerBatting] = useState({ headers: [], values: [] });

  useEffect(() => {
    const fetchTrendingPlayers = async () => {
      try {
        const response = await axios.get('http://localhost:3001/api/trending-players');
        setTrendingPlayers(response.data);
      } catch (error) {
        console.error('Error fetching trending players:', error);
      }
    };

    fetchTrendingPlayers();
  }, []);

  useEffect(() => {
    if (selectedPlayerId) {
      const fetchPlayerData = async () => {
        try {
          const careerResponse = await axios.get(`http://localhost:3001/api/player/${selectedPlayerId}/career`);
          const bowlingResponse = await axios.get(`http://localhost:3001/api/player/${selectedPlayerId}/bowling`);
          const battingResponse = await axios.get(`http://localhost:3001/api/player/${selectedPlayerId}/batting`);

          setPlayerCareer(careerResponse.data);
          setPlayerBowling(bowlingResponse.data);
          setPlayerBatting(battingResponse.data);
        } catch (error) {
          console.error('Error fetching player data:', error);
        }
      };

      fetchPlayerData();
    }
  }, [selectedPlayerId]);

  return (
    <div className="App">
      <h1>Trending Cricket Players</h1>
      <select onChange={(e) => setSelectedPlayerId(e.target.value)}>
        <option value="">Select a player</option>
        {trendingPlayers.map((player) => (
          <option key={player.id} value={player.id}>
            {player.name} - {player.teamName}
          </option>
        ))}
      </select>

      {selectedPlayerId && (
        <div className="player-stats">
          <h2>Career</h2>
          <ul>
            {playerCareer.map((stat, index) => (
              <li key={index}>
                {stat.name}: {stat.debut} - {stat.lastPlayed}
              </li>
            ))}
          </ul>

          <h2>Bowling</h2>
          <table>
            <thead>
              <tr>
                {playerBowling.headers && playerBowling.headers.map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {playerBowling.values && Array.isArray(playerBowling.values) && playerBowling.values.map((row, index) => (
                <tr key={index}>
                  {row.values.map((value, i) => (
                    <td key={i}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          <h2>Batting</h2>
          <table>
            <thead>
              <tr>
                {playerBatting.headers && playerBatting.headers.map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {playerBatting.values && Array.isArray(playerBatting.values) && playerBatting.values.map((row, index) => (
                <tr key={index}>
                  {row.values.map((value, i) => (
                    <td key={i}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default App;
