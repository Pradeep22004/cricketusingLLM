import React from 'react';
import './Header.css';
const Sidebar = () => {
  return (
    <div className="sidebar">
      <ul>
        <li><a href="./card">Players</a></li>
        <li><a href="#">Match Reports</a></li>
        <li><a href="#">Track Records</a></li>
        
        {/* Add more navigation links as needed */}
      </ul>
    </div>
  );
};

export default Sidebar;