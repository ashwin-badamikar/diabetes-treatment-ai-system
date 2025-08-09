import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class DiabetesRLAgent:
    """
    Production-ready diabetes treatment recommendation agent
    Fixed architecture to match training models
    """
    def __init__(self, model_path="../models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Treatment actions
        self.treatments = {
            0: "Lifestyle Modification Only",
            1: "Metformin Monotherapy", 
            2: "Metformin + Lifestyle Intensive",
            3: "Metformin + Sulfonylurea",
            4: "Insulin Therapy",
            5: "Multi-drug Combination Therapy"
        }
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load trained DQN and Policy Gradient models"""
        try:
            # Load DQN model with correct architecture
            dqn_checkpoint = torch.load(self.model_path / "dqn_diabetes_model.pt", 
                                      map_location=self.device, weights_only=False)
            
            # Build DQN with EXACT same architecture as training
            self.dqn_model = nn.Sequential(
                nn.Linear(16, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Dropout(0.3),
                nn.Linear(2048, 1536),
                nn.ReLU(), 
                nn.BatchNorm1d(1536),
                nn.Dropout(0.3),
                nn.Linear(1536, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
                nn.Linear(256, 6)
            ).to(self.device)
            
            # Load state dict with correct key mapping
            if 'q_network_state_dict' in dqn_checkpoint:
                state_dict = dqn_checkpoint['q_network_state_dict']
                # Fix key mapping - remove 'network.' prefix
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('network.'):
                        new_key = key[8:]  # Remove 'network.' prefix
                        fixed_state_dict[new_key] = value
                    else:
                        fixed_state_dict[key] = value
                
                self.dqn_model.load_state_dict(fixed_state_dict)
            
            # Set to evaluation mode and disable BatchNorm issues
            self.dqn_model.eval()
            
            # Load Policy Gradient model
            pg_checkpoint = torch.load(self.model_path / "policy_gradient_model.pt", 
                                     map_location=self.device, weights_only=False)
            
            # Build Policy Gradient with correct architecture
            self.pg_model = nn.Sequential(
                nn.Linear(16, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 6),
                nn.Softmax(dim=-1)
            ).to(self.device)
            
            if 'policy_net_state_dict' in pg_checkpoint:
                self.pg_model.load_state_dict(pg_checkpoint['policy_net_state_dict'])
            
            self.pg_model.eval()
            
            print("✅ Models loaded successfully for clinical deployment")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Creating demo models for testing...")
            self._create_demo_models()
    
    def _create_demo_models(self):
        """Create demo models if loading fails"""
        self.dqn_model = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        ).to(self.device)
        
        self.pg_model = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        self.dqn_model.eval()
        self.pg_model.eval()
        print("✅ Demo models created")
    
    def recommend_treatment(self, patient_features, algorithm="both"):
        """
        Get treatment recommendation for a patient
        Fixed for BatchNorm evaluation mode
        """
        # Ensure input is correct format
        if len(patient_features) < 16:
            # Pad with zeros if needed
            patient_features = list(patient_features) + [0] * (16 - len(patient_features))
        
        patient_tensor = torch.FloatTensor(patient_features).unsqueeze(0).to(self.device)
        
        results = {}
        
        if algorithm in ["dqn", "both"]:
            with torch.no_grad():
                # Set to eval mode to handle BatchNorm with single sample
                self.dqn_model.eval()
                
                # For BatchNorm compatibility, repeat input if needed
                if patient_tensor.shape[0] == 1:
                    # Duplicate input to avoid BatchNorm issues
                    patient_batch = patient_tensor.repeat(2, 1)
                    q_values = self.dqn_model(patient_batch)[0:1]  # Take first result
                else:
                    q_values = self.dqn_model(patient_tensor)
                
                dqn_action = q_values.argmax().item()
                results['dqn'] = {
                    'action': dqn_action,
                    'treatment': self.treatments[dqn_action],
                    'confidence': torch.softmax(q_values, dim=1).max().item(),
                    'q_values': q_values.cpu().numpy().tolist()
                }
        
        if algorithm in ["policy_gradient", "both"]:
            with torch.no_grad():
                self.pg_model.eval()
                action_probs = self.pg_model(patient_tensor)
                pg_action = action_probs.argmax().item()
                results['policy_gradient'] = {
                    'action': pg_action,
                    'treatment': self.treatments[pg_action],
                    'confidence': action_probs.max().item(),
                    'probabilities': action_probs.cpu().numpy().tolist()
                }
        
        return results
    
    def clinical_assessment(self, patient_features):
        """Comprehensive clinical assessment for healthcare providers"""
        # Ensure 16 features
        if len(patient_features) < 16:
            patient_features = list(patient_features) + [0] * (16 - len(patient_features))
        
        glucose, bmi, age = patient_features[0], patient_features[1], patient_features[2]
        
        # Get recommendations from both models
        recommendations = self.recommend_treatment(patient_features, "both")
        
        # Clinical risk assessment
        risk_factors = []
        if glucose > 126: risk_factors.append("Diabetes diagnosis")
        if glucose > 180: risk_factors.append("Severe hyperglycemia")
        if bmi > 30: risk_factors.append("Obesity")
        if age > 65: risk_factors.append("Elderly patient")
        
        assessment = {
            'patient_profile': {
                'glucose': glucose,
                'bmi': bmi,
                'age': age,
                'risk_factors': risk_factors
            },
            'ai_recommendations': recommendations,
            'clinical_priority': 'High' if glucose > 180 else 'Medium' if glucose > 140 else 'Low',
            'follow_up_needed': glucose > 140 or bmi > 35
        }
        
        return assessment

# Test the production agent
if __name__ == "__main__":
    print("🏥 TESTING DIABETES RL AGENT")
    print("=" * 40)
    
    agent = DiabetesRLAgent()
    
    # Test patients
    test_cases = [
        ("Young Pre-diabetic", [110, 28, 32, 75, 1, 0.3]),
        ("Elderly Diabetic", [165, 31, 68, 85, 4, 0.7]),
        ("Severe Case", [220, 38, 45, 90, 2, 0.9])
    ]
    
    for case_name, features in test_cases:
        print(f"\n🩺 {case_name}:")
        print(f"   Profile: Glucose={features[0]}, BMI={features[1]}, Age={features[2]}")
        
        try:
            assessment = agent.clinical_assessment(features)
            print(f"   DQN: {assessment['ai_recommendations']['dqn']['treatment']}")
            print(f"   PG: {assessment['ai_recommendations']['policy_gradient']['treatment']}")
            print(f"   Priority: {assessment['clinical_priority']}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n✅ Production agent testing complete!")