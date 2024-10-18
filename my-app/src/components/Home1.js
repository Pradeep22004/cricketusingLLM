import React from "react";
import { useLocation } from 'react-router-dom';

function Home1() {
    const location = useLocation();
    // const userId = location.state ? location.state.id : "Guest";

    const styles = {
        homepage: {
            height: "100vh",
            width: "100vw",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            overflow: "hidden"
        },
        homepageImage: {
            height: "100%",
            width: "100%",
            objectFit: "cover"
        }
    };

    return (
        <div style={styles.homepage}>
            <img src="/1.jpg" alt="cricnet" style={styles.homepageImage} />
        </div>
    );
}

export default Home1;
