import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './Schedules.css';

const Schedule = () => {
  const [schedule, setSchedule] = useState([]);

  useEffect(() => {
    const fetchSchedule = async () => {
      try {
        const response = await axios.get('/api/schedule');
        setSchedule(response.data.matchScheduleMap);
      } catch (error) {
        console.error('Error fetching schedule data:', error);
      }
    };

    fetchSchedule();
  }, []);

  return (
    <div className="container">
      <h1 className="heading">Cricket Schedule</h1>
      {schedule.map((scheduleWrapper, index) => (
        <div key={index}>
          {scheduleWrapper.scheduleAdWrapper && (
            <div>
              <h2 className="sub-heading">{scheduleWrapper.scheduleAdWrapper.date}</h2>
              <table>
                <thead>
                  <tr>
                    <th>Match Description</th>
                    <th>Teams</th>
                    <th>Venue</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {scheduleWrapper.scheduleAdWrapper.matchScheduleList.map((match, matchIndex) => (
                    match.matchInfo.map((info, infoIndex) => (
                      <tr key={info.matchId}>
                        <td>{info.matchDesc}</td>
                        <td>{info.team1.teamName} vs {info.team2.teamName}</td>
                        <td>{info.venueInfo.ground}, {info.venueInfo.city}</td>
                        <td>{new Date(parseInt(info.startDate)).toLocaleString()}</td>
                      </tr>
                    ))
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default Schedule;

