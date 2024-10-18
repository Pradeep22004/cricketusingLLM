// App.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './StatsRecords.css';

const StatsRecords = () => {
  const [filters, setFilters] = useState([]);
  const [records, setRecords] = useState([]);
  const [selectedStatsType, setSelectedStatsType] = useState('mostRuns');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchFilters = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/api/stats/get-record-filters');
        setFilters(response.data.statsTypesList);
        setLoading(false);
      } catch (error) {
        setLoading(false);
        setError('Error fetching filters');
        console.error('Error fetching filters:', error);
      }
    };

    fetchFilters();
  }, []);

  useEffect(() => {
    const fetchRecords = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/api/stats/get-records', {
          params: { statsType: selectedStatsType }
        });
        setRecords(response.data.values);
        setLoading(false);
      } catch (error) {
        setLoading(false);
        setError('Error fetching records');
        console.error('Error fetching records:', error);
      }
    };

    fetchRecords();
  }, [selectedStatsType]);

  return (
    <div className="container">
      <h1>Cricket Stats</h1>
      <div>
        <h2>Select Stat Type</h2>
        <select
          onChange={(e) => setSelectedStatsType(e.target.value)}
          value={selectedStatsType}
        >
          {filters.map((category) =>
            category.types.map((type) => (
              <option key={type.value} value={type.value}>
                {type.header}
              </option>
            ))
          )}
        </select>
      </div>
      <div>
        <h2>Records</h2>
        {loading ? (
          <p>Loading...</p>
        ) : error ? (
          <p>{error}</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Player ID</th>
                <th>Player</th>
                <th>Matches</th>
                <th>Innings</th>
                <th>Runs</th>
                <th>Avg</th>
              </tr>
            </thead>
            <tbody>
              {records.map((record, index) => (
                <tr key={index}>
                  {record.values.map((value, i) => (
                    <td key={i}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {records.length > 0 && (
        <div className="stats-box">
          <h2>Stats Summary</h2>
          <p>Total Players: {records.length}</p>
          <p>Top Player: {records[0].values[1]}</p>
          <p>Top Runs: {records[0].values[4]}</p>
        </div>
      )}
    </div>
  );
};

export default StatsRecords;
