import React from 'react';
import './PlayerCard.css'; // Import CSS file for styling
// import './kohli.png';
// import './rohit.png';
// import './bumrah.png';
import rohitPhoto from './rohit.png'; // Import player photos
import kohliPhoto from './kohli.png';
import bumrahPhoto from './bumrah.png';
import jadejaPhoto from './jadeja.png';
import ashwinPhoto from './ashwin.png';
import shamiPhoto from './shami.png';
import sirajPhoto from './siraj.png';
import rahulPhoto from './rahul.png';
import gillPhoto from './gill.png';
import pandyaPhoto from './pandya.png';

const PlayerCard = ({ player }) => {
  const { name, photo, battingStyle, age, contribution } = player;
  
  let playerPhoto;
  // Assign the correct player photo based on the player's name
  switch (name) {
    case 'Rohit Sharma':
      playerPhoto = rohitPhoto;
      break;
    case 'Virat Kohli':
      playerPhoto = kohliPhoto;
      break;
    case 'Jasprit Bumrah':
      playerPhoto = bumrahPhoto;
      break;
      case 'Ravindra Jadeja':
      playerPhoto = jadejaPhoto;
      break;
      case 'Ravichandran Ashwin':
        playerPhoto = ashwinPhoto;
        break;
        case 'Mohammad Shami':
          playerPhoto = shamiPhoto;
          break; 
          case 'Mohammed Siraj':
            playerPhoto = sirajPhoto;
            break; 
            case 'KL Rahul':
              playerPhoto = rahulPhoto;
              break;
              case 'Shubman Gill':
                playerPhoto = gillPhoto;
                break;
                case 'Hardik Pandya':
                  playerPhoto = pandyaPhoto;
                  break;
    default:
      playerPhoto = null; // Set a default photo if the player name doesn't match
  }

  return (
    <a href="#" className="player-card"> {/* Wrap the card in an anchor tag */}
      <div className="player-photo">
        <img src={playerPhoto} alt={name} />
      </div>
      <div className="player-details">
        <div className="player-info">
          <h2>{name}</h2>
          <p>Batting Style: {battingStyle}</p>
          <p>Age: {age}</p>
        </div>
        <div className="player-contribution">
          <p>{contribution}</p>
        </div>
      </div>
    </a>
  );
};

export default PlayerCard;