// import React, { useState, useEffect } from 'react';
// import axios from 'axios';

// const News = () => {
//   const [newsList, setNewsList] = useState([]);
//   const [selectedNews, setSelectedNews] = useState(null);
//   const [error, setError] = useState('');

//   useEffect(() => {
//     const fetchNewsList = async () => {
//       try {
//         const response = await axios.get('http://localhost:3000/api/news/list');
//         setNewsList(response.data);
//         setError('');
//       } catch (err) {
//         console.error('Error fetching news list:', err);
//         setError('Failed to fetch news list');
//       }
//     };

//     fetchNewsList();
//   }, []);

//   const fetchNewsDetails = async (newsId) => {
//     try {
//       const response = await axios.get(`http://localhost:3000/api/news/detail/${newsId}`);
//       setSelectedNews(response.data);
//       setError('');
//     } catch (err) {
//       console.error('Error fetching news details:', err);
//       setError('Failed to fetch news details');
//     }
//   };

//   return (
//     <div style={styles.container}>
//       <h1>News</h1>
//       {error && <p style={styles.error}>{error}</p>}
//       {selectedNews ? (
//         <div style={styles.newsDetails}>
//           <button onClick={() => setSelectedNews(null)} style={styles.backButton}>Back to News List</button>
//           <h2>{selectedNews.story.hline}</h2>
//           <p>{selectedNews.story.intro}</p>
//         </div>
//       ) : (
//         <ul style={styles.newsList}>
//           {newsList.map((news) => (
//             <li key={news.story.id} style={styles.newsItem}>
//               <h3>{news.story.hline}</h3>
//               <p>{news.story.intro}</p>
//               <button onClick={() => fetchNewsDetails(news.story.id)} style={styles.button}>Read More</button>
//             </li>
//           ))}
//         </ul>
//       )}
//     </div>
//   );
// };

// const styles = {
//   container: {
//     fontFamily: 'Arial, sans-serif',
//     padding: '20px',
//     maxWidth: '800px',
//     margin: '0 auto',
//     backgroundColor: '#f4f4f9',
//     borderRadius: '8px',
//     boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
//   },
//   error: {
//     color: 'red',
//   },
//   newsList: {
//     listStyleType: 'none',
//     padding: '0',
//   },
//   newsItem: {
//     marginBottom: '20px',
//     padding: '10px',
//     backgroundColor: '#fff',
//     borderRadius: '4px',
//     boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
//   },
//   newsDetails: {
//     padding: '20px',
//     backgroundColor: '#fff',
//     borderRadius: '4px',
//     boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
//   },
//   button: {
//     padding: '10px 20px',
//     fontSize: '16px',
//     borderRadius: '4px',
//     border: 'none',
//     backgroundColor: '#007BFF',
//     color: '#fff',
//     cursor: 'pointer',
//     transition: 'background-color 0.3s ease',
//   },
//   backButton: {
//     padding: '10px 20px',
//     fontSize: '16px',
//     borderRadius: '4px',
//     border: 'none',
//     backgroundColor: '#6c757d',
//     color: '#fff',
//     cursor: 'pointer',
//     transition: 'background-color 0.3s ease',
//     marginBottom: '20px',
//   },
// };

// export default News;


// import React, { useState, useEffect } from 'react';
// import axios from 'axios';

// const News = () => {
//   const [newsList, setNewsList] = useState([]);
//   const [selectedNews, setSelectedNews] = useState(null);
//   const [error, setError] = useState('');

//   useEffect(() => {
//     const fetchNewsList = async () => {
//       try {
//         const response = await axios.get('http://localhost:3000/api/news/list');
//         setNewsList(response.data);
//         setError('');
//       } catch (err) {
//         console.error('Error fetching news list:', err);
//         setError('Failed to fetch news list');
//       }
//     };

//     fetchNewsList();
//   }, []);

//   const fetchNewsDetails = async (newsId) => {
//     try {
//       const response = await axios.get(`http://localhost:3000/api/news/detail/${newsId}`);
//       console.log('News details response:', response.data); // Log the response data
//       setSelectedNews(response.data);
//       setError('');
//     } catch (err) {
//       console.error('Error fetching news details:', err);
//       setError('Failed to fetch news details');
//     }
//   };

//   return (
//     <div style={styles.container}>
//       <h1>News</h1>
//       {error && <p style={styles.error}>{error}</p>}
//       {selectedNews ? (
//         <div style={styles.newsDetails}>
//           <button onClick={() => setSelectedNews(null)} style={styles.backButton}>Back to News List</button>
//           {selectedNews.story ? ( // Check if selectedNews.story exists
//             <>
//               <h2>{selectedNews.story.hline}</h2>
//               <p>{selectedNews.story.intro}</p>
//             </>
//           ) : (
//             <p>Loading...</p> // Display a loading message or an appropriate message
//           )}
//         </div>
//       ) : (
//         <ul style={styles.newsList}>
//           {newsList.map((news) => (
//             <li key={news.story.id} style={styles.newsItem}>
//               <h3>{news.story.hline}</h3>
//               <p>{news.story.intro}</p>
//               <button onClick={() => fetchNewsDetails(news.story.id)} style={styles.button}>Read More</button>
//             </li>
//           ))}
//         </ul>
//       )}
//     </div>
//   );
// };

// const styles = {
//   container: {
//     fontFamily: 'Arial, sans-serif',
//     padding: '20px',
//     maxWidth: '800px',
//     margin: '0 auto',
//     backgroundColor: '#f4f4f9',
//     borderRadius: '8px',
//     boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
//   },
//   error: {
//     color: 'red',
//   },
//   newsList: {
//     listStyleType: 'none',
//     padding: '0',
//   },
//   newsItem: {
//     marginBottom: '20px',
//     padding: '10px',
//     backgroundColor: '#fff',
//     borderRadius: '4px',
//     boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
//   },
//   newsDetails: {
//     padding: '20px',
//     backgroundColor: '#fff',
//     borderRadius: '4px',
//     boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
//   },
//   button: {
//     padding: '10px 20px',
//     fontSize: '16px',
//     borderRadius: '4px',
//     border: 'none',
//     backgroundColor: '#007BFF',
//     color: '#fff',
//     cursor: 'pointer',
//     transition: 'background-color 0.3s ease',
//   },
//   backButton: {
//     padding: '10px 20px',
//     fontSize: '16px',
//     borderRadius: '4px',
//     border: 'none',
//     backgroundColor: '#6c757d',
//     color: '#fff',
//     cursor: 'pointer',
//     transition: 'background-color 0.3s ease',
//     marginBottom: '20px',
//   },
// };

// export default News;



import React, { useState, useEffect } from 'react';
import axios from 'axios';

const News = () => {
  const [newsList, setNewsList] = useState([]);
  const [selectedNews, setSelectedNews] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchNewsList = async () => {
      try {
        const response = await axios.get('http://localhost:3000/api/news/list');
        setNewsList(response.data); // Adjust based on actual response structure
        setError('');
      } catch (err) {
        console.error('Error fetching news list:', err);
        setError('Failed to fetch news list');
      }
    };

    fetchNewsList();
  }, []);

  const fetchNewsDetails = async (newsId) => {
    try {
      const response = await axios.get(`http://localhost:3000/api/news/detail/${newsId}`);
      console.log('News details response:', response.data); // Log the response data
      setSelectedNews(response.data);
      setError('');
    } catch (err) {
      console.error('Error fetching news details:', err);
      setError('Failed to fetch news details');
    }
  };

  return (
    <div style={styles.container}>
      <h1>News</h1>
      {error && <p style={styles.error}>{error}</p>}
      {selectedNews ? (
        <div style={styles.newsDetails}>
          <button onClick={() => setSelectedNews(null)} style={styles.backButton}>Back to News List</button>
          {selectedNews.headline ? ( // Check if selectedNews.headline exists
            <>
              <h2>{selectedNews.headline}</h2>
              {selectedNews.coverImage && (
                <img src={`https://cricbuzz-cricket.p.rapidapi.com/img/v1/i1/c${selectedNews.coverImage.id}.jpg`} alt={selectedNews.coverImage.caption} />
              )}
              <p>{selectedNews.intro}</p>
              {selectedNews.content.map((contentItem, index) => (
                contentItem.content && contentItem.content.contentValue ? (
                <p key={index}>{contentItem.content.contentValue}</p>
              ) : null
              ))}
            </>
          ) : (
            <p>Loading...</p> // Display a loading message or an appropriate message
          )}
        </div>
      ) : (
        <ul style={styles.newsList}>
          {newsList.map((news) => (
            <li key={news.story.id} style={styles.newsItem}>
              <h3>{news.story.hline}</h3>
              <p>{news.story.intro}</p>
              <button onClick={() => fetchNewsDetails(news.story.id)} style={styles.button}>Read More</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

const styles = {
  container: {
    fontFamily: 'Arial, sans-serif',
    padding: '20px',
    maxWidth: '800px',
    margin: '0 auto',
    backgroundColor: '#c7f3ff',
    borderRadius: '8px',
    boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
  },
  error: {
    color: 'red',
  },
  newsList: {
    listStyleType: 'none',
    padding: '0',
  },
  newsItem: {
    marginBottom: '20px',
    padding: '10px',
    backgroundColor: '#fffdfb',
    borderRadius: '4px',
    boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
  },
  newsDetails: {
    padding: '20px',
    backgroundColor: '#fff',
    borderRadius: '4px',
    boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)',
  },
  button: {
    padding: '10px 20px',
    fontSize: '16px',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: '#01005e',
    color: '#fff',
    cursor: 'pointer',
    transition: 'background-color 0.3s ease',
  },
  backButton: {
    padding: '10px 20px',
    fontSize: '16px',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: '#6c757d',
    color: '#fff',
    cursor: 'pointer',
    transition: 'background-color 0.3s ease',
    marginBottom: '20px',
  },
};

export default News;
