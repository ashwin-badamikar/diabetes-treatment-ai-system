import React, { useState } from 'react';
import './App.css';

function App() {
  const [patientData, setPatientData] = useState({
    glucose: '',
    bmi: '',
    age: '',
    blood_pressure: '',
    pregnancies: '',
    diabetes_pedigree: ''
  });
  
  const [recommendation, setRecommendation] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch('http://localhost:8000/recommend_treatment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData)
      });
      
      const result = await response.json();
      setRecommendation(result);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏥 Diabetes Treatment AI System</h1>
        <p>AI-Powered Treatment Recommendations</p>
      </header>
      
      <div className="patient-form">
        <h2>Patient Information</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="number"
            placeholder="Glucose Level (mg/dL)"
            value={patientData.glucose}
            onChange={(e) => setPatientData({...patientData, glucose: e.target.value})}
          />
          <input
            type="number"
            placeholder="BMI"
            value={patientData.bmi}
            onChange={(e) => setPatientData({...patientData, bmi: e.target.value})}
          />
          <input
            type="number"
            placeholder="Age"
            value={patientData.age}
            onChange={(e) => setPatientData({...patientData, age: e.target.value})}
          />
          <button type="submit">Get AI Recommendation</button>
        </form>
      </div>
      
      {recommendation && (
        <div className="recommendations">
          <h2>🤖 AI Treatment Recommendations</h2>
          <div className="dqn-recommendation">
            <h3>DQN Algorithm:</h3>
            <p>{recommendation.recommendations.dqn.treatment}</p>
            <p>Confidence: {(recommendation.recommendations.dqn.confidence * 100).toFixed(1)}%</p>
          </div>
          <div className="pg-recommendation">
            <h3>Policy Gradient Algorithm:</h3>
            <p>{recommendation.recommendations.policy_gradient.treatment}</p>
            <p>Confidence: {(recommendation.recommendations.policy_gradient.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;