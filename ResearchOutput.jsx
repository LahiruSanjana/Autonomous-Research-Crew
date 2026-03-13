import React, { useState } from "react";

function ResearchOutput() {
  const [output, setOutput] = useState("");

  // This would be replaced by an API call in a real app
  const handleShowOutput = () => {
    // Example output, replace with actual backend call
    setOutput(`\
=== Final Research Report ===\n\n1. Energy-efficient IoT protocols and architectures 2020–2023: A comparative analysis of LoRaWAN, NB-IoT, Wi-SUN, and other low-power wide-area network (LPWAN) and edge-based solutions for extended battery life and reduced power consumption in IoT devices\n2. Hardware-level innovations in ultra-low-power IoT systems: State-of-the-art microcontrollers (e.g., nRF52, ESP32, SAM L11), power management ICs (PMICs), sleep modes, and harvesting techniques for wireless sensor networks (WSNs) and embedded applications\n3. Machine learning and AI for optimizing energy efficiency in IoT systems: Recent studies (2021–2024) on predictive power management, dynamic duty cycling, adaptive duty cycles, and edge/device optimization algorithms for real-time energy reduction in smart grids, industrial automation, and wearable devices\n`);
  };

  return (
    <div style={{ maxWidth: 700, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h2>Research Crew Output Viewer</h2>
      <button onClick={handleShowOutput} style={{ padding: "10px 20px", fontSize: 16 }}>
        Show Final Output
      </button>
      {output && (
        <pre style={{ background: "#f4f4f4", padding: 20, marginTop: 20, borderRadius: 8 }}>
          {output}
        </pre>
      )}
    </div>
  );
}

export default ResearchOutput;
