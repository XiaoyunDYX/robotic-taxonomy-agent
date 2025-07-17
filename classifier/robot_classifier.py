import json
import re
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class RobotTaxonomy:
    def __init__(self):
        # Define the hierarchical taxonomy structure
        self.taxonomy = {
            "Domain": {
                "Industrial": {
                    "Manufacturing": ["Assembly", "Welding", "Painting", "Material Handling", "Quality Control"],
                    "Logistics": ["Warehouse", "Sorting", "Packaging", "Transportation"],
                    "Construction": ["Demolition", "Building", "Inspection", "Maintenance"]
                },
                "Service": {
                    "Healthcare": ["Surgical", "Rehabilitation", "Diagnostic", "Patient Care", "Pharmacy"],
                    "Domestic": ["Cleaning", "Cooking", "Security", "Entertainment", "Companion"],
                    "Commercial": ["Retail", "Hospitality", "Education", "Office"]
                },
                "Research": {
                    "Exploration": ["Space", "Underwater", "Aerial", "Terrestrial"],
                    "Laboratory": ["Laboratory", "Research", "Experimental"],
                    "Educational": ["Teaching", "Learning", "Training"]
                },
                "Military": {
                    "Combat": ["Weapons", "Reconnaissance", "Bomb Disposal"],
                    "Support": ["Logistics", "Medical", "Engineering"],
                    "Surveillance": ["Reconnaissance", "Monitoring", "Intelligence"]
                }
            },
            "Mobility": {
                "Stationary": ["Fixed", "Mounted", "Permanent"],
                "Mobile": {
                    "Wheeled": ["Differential Drive", "Omnidirectional", "Tracked"],
                    "Legged": ["Bipedal", "Quadrupedal", "Hexapod", "Multi-legged"],
                    "Flying": ["Fixed-wing", "Rotary-wing", "Multi-rotor"],
                    "Swimming": ["Underwater", "Surface", "Hybrid"]
                }
            },
            "Autonomy": {
                "Teleoperated": ["Remote Control", "Human-in-the-loop"],
                "Semi-autonomous": ["Supervised", "Assisted"],
                "Autonomous": ["Fully Autonomous", "AI-driven", "Self-learning"]
            },
            "Size": {
                "Nano": ["Microscopic", "Molecular"],
                "Micro": ["Millimeter", "Centimeter"],
                "Small": ["Handheld", "Portable"],
                "Medium": ["Human-sized", "Table-top"],
                "Large": ["Industrial", "Vehicle-sized"],
                "Macro": ["Building-sized", "Infrastructure"]
            }
        }
        
        # Keywords for classification
        self.keywords = self._extract_keywords()
        
    def _extract_keywords(self) -> Dict[str, List[str]]:
        """
        Extract keywords from taxonomy for classification
        """
        keywords = {}
        
        def extract_from_dict(d, prefix=""):
            for key, value in d.items():
                current_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    extract_from_dict(value, current_key)
                elif isinstance(value, list):
                    keywords[current_key] = value
                else:
                    keywords[current_key] = [value]
        
        extract_from_dict(self.taxonomy)
        return keywords
    
    def classify_robot(self, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a robot according to the taxonomy
        """
        classification = {
            "Domain": {},
            "Mobility": {},
            "Autonomy": {},
            "Size": {}
        }
        
        # Extract text for analysis
        text = self._extract_text(robot_data)
        text_lower = text.lower()
        
        # Classify by domain
        classification["Domain"] = self._classify_domain(text_lower, robot_data)
        
        # Classify by mobility
        classification["Mobility"] = self._classify_mobility(text_lower, robot_data)
        
        # Classify by autonomy
        classification["Autonomy"] = self._classify_autonomy(text_lower, robot_data)
        
        # Classify by size
        classification["Size"] = self._classify_size(text_lower, robot_data)
        
        return classification
    
    def _extract_text(self, robot_data: Dict[str, Any]) -> str:
        """
        Extract all text from robot data for analysis
        """
        text_parts = []
        
        if 'name' in robot_data:
            text_parts.append(robot_data['name'])
        
        if 'description' in robot_data:
            text_parts.append(robot_data['description'])
        
        if 'applications' in robot_data:
            text_parts.extend(robot_data['applications'])
        
        if 'category' in robot_data:
            text_parts.append(robot_data['category'])
        
        return ' '.join(text_parts)
    
    def _classify_domain(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by domain
        """
        domain_scores = {}
        
        # Check for industrial keywords
        industrial_keywords = ['industrial', 'manufacturing', 'factory', 'assembly', 'welding', 'painting']
        if any(keyword in text for keyword in industrial_keywords):
            domain_scores['Industrial'] = 0.8
        
        # Check for service keywords
        service_keywords = ['service', 'domestic', 'home', 'cleaning', 'cooking', 'companion']
        if any(keyword in text for keyword in service_keywords):
            domain_scores['Service'] = 0.8
        
        # Check for research keywords
        research_keywords = ['research', 'experimental', 'laboratory', 'exploration', 'space', 'underwater']
        if any(keyword in text for keyword in research_keywords):
            domain_scores['Research'] = 0.8
        
        # Check for military keywords
        military_keywords = ['military', 'defense', 'weapon', 'combat', 'surveillance', 'reconnaissance']
        if any(keyword in text for keyword in military_keywords):
            domain_scores['Military'] = 0.8
        
        # If no clear domain, default to service
        if not domain_scores:
            domain_scores['Service'] = 0.5
        
        return domain_scores
    
    def _classify_mobility(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by mobility type
        """
        mobility_scores = {}
        
        # Check for stationary keywords
        stationary_keywords = ['fixed', 'mounted', 'stationary', 'permanent']
        if any(keyword in text for keyword in stationary_keywords):
            mobility_scores['Stationary'] = 0.8
        
        # Check for mobile keywords
        mobile_keywords = ['mobile', 'moving', 'wheeled', 'legged', 'flying', 'swimming']
        if any(keyword in text for keyword in mobile_keywords):
            mobility_scores['Mobile'] = 0.8
            
            # Sub-classify mobile robots
            if any(word in text for word in ['wheel', 'drive', 'track']):
                mobility_scores['Mobile.Wheeled'] = 0.7
            elif any(word in text for word in ['leg', 'biped', 'quadruped', 'walk']):
                mobility_scores['Mobile.Legged'] = 0.7
            elif any(word in text for word in ['fly', 'drone', 'aerial', 'helicopter']):
                mobility_scores['Mobile.Flying'] = 0.7
            elif any(word in text for word in ['swim', 'underwater', 'submarine']):
                mobility_scores['Mobile.Swimming'] = 0.7
        
        # Default to mobile if unclear
        if not mobility_scores:
            mobility_scores['Mobile'] = 0.5
        
        return mobility_scores
    
    def _classify_autonomy(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by autonomy level
        """
        autonomy_scores = {}
        
        # Check for teleoperated keywords
        teleop_keywords = ['remote', 'teleoperated', 'controlled', 'manual']
        if any(keyword in text for keyword in teleop_keywords):
            autonomy_scores['Teleoperated'] = 0.8
        
        # Check for autonomous keywords
        autonomous_keywords = ['autonomous', 'ai', 'artificial intelligence', 'self-driving', 'independent']
        if any(keyword in text for keyword in autonomous_keywords):
            autonomy_scores['Autonomous'] = 0.8
        
        # Check for semi-autonomous keywords
        semi_keywords = ['semi', 'assisted', 'supervised', 'collaborative']
        if any(keyword in text for keyword in semi_keywords):
            autonomy_scores['Semi-autonomous'] = 0.8
        
        # Default to semi-autonomous if unclear
        if not autonomy_scores:
            autonomy_scores['Semi-autonomous'] = 0.5
        
        return autonomy_scores
    
    def _classify_size(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by size
        """
        size_scores = {}
        
        # Check for size keywords
        size_keywords = {
            'Nano': ['nano', 'microscopic', 'molecular'],
            'Micro': ['micro', 'millimeter', 'centimeter'],
            'Small': ['small', 'handheld', 'portable', 'mini'],
            'Medium': ['medium', 'human', 'table'],
            'Large': ['large', 'industrial', 'vehicle'],
            'Macro': ['macro', 'building', 'infrastructure']
        }
        
        for size, keywords in size_keywords.items():
            if any(keyword in text for keyword in keywords):
                size_scores[size] = 0.8
        
        # Default to medium if unclear
        if not size_scores:
            size_scores['Medium'] = 0.5
        
        return size_scores

class RobotClassifier:
    def __init__(self):
        self.taxonomy = RobotTaxonomy()
        self.classified_robots = []
        
    def classify_robots(self, robots_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a list of robots according to the taxonomy
        """
        classified_robots = []
        
        for robot in robots_data:
            classification = self.taxonomy.classify_robot(robot)
            robot['classification'] = classification
            classified_robots.append(robot)
            
        self.classified_robots = classified_robots
        return classified_robots
    
    def cluster_robots(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Use machine learning to cluster robots based on their descriptions
        """
        if not self.classified_robots:
            return {}
        
        # Extract text features
        texts = [robot.get('description', '') + ' ' + robot.get('name', '') 
                for robot in self.classified_robots]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster information to robots
        for i, robot in enumerate(self.classified_robots):
            robot['cluster'] = int(clusters[i])
        
        return {
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'feature_names': vectorizer.get_feature_names_out().tolist()
        }
    
    def save_classified_data(self, filename: str = 'classified_robots.json'):
        """
        Save classified robot data to JSON file
        """
        with open(f'./data/{filename}', 'w') as f:
            json.dump(self.classified_robots, f, indent=2)
        print(f"Saved {len(self.classified_robots)} classified robots to {filename}")
    
    def get_taxonomy_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the classification results
        """
        summary = {
            'total_robots': len(self.classified_robots),
            'domain_distribution': {},
            'mobility_distribution': {},
            'autonomy_distribution': {},
            'size_distribution': {}
        }
        
        for robot in self.classified_robots:
            classification = robot.get('classification', {})
            
            # Count domain classifications
            for domain, score in classification.get('Domain', {}).items():
                if score > 0.5:  # Only count confident classifications
                    summary['domain_distribution'][domain] = summary['domain_distribution'].get(domain, 0) + 1
            
            # Count mobility classifications
            for mobility, score in classification.get('Mobility', {}).items():
                if score > 0.5:
                    summary['mobility_distribution'][mobility] = summary['mobility_distribution'].get(mobility, 0) + 1
            
            # Count autonomy classifications
            for autonomy, score in classification.get('Autonomy', {}).items():
                if score > 0.5:
                    summary['autonomy_distribution'][autonomy] = summary['autonomy_distribution'].get(autonomy, 0) + 1
            
            # Count size classifications
            for size, score in classification.get('Size', {}).items():
                if score > 0.5:
                    summary['size_distribution'][size] = summary['size_distribution'].get(size, 0) + 1
        
        return summary

if __name__ == "__main__":
    # Load robot data
    try:
        with open('./data/robots_data.json', 'r') as f:
            robots_data = json.load(f)
    except FileNotFoundError:
        print("No robot data found. Please run the scraper first.")
        robots_data = []
    
    if robots_data:
        classifier = RobotClassifier()
        classified_robots = classifier.classify_robots(robots_data)
        
        # Perform clustering
        clustering_info = classifier.cluster_robots()
        
        # Save results
        classifier.save_classified_data()
        
        # Print summary
        summary = classifier.get_taxonomy_summary()
        print(f"Classified {summary['total_robots']} robots")
        print("Domain distribution:", summary['domain_distribution'])
        print("Mobility distribution:", summary['mobility_distribution'])
        print("Autonomy distribution:", summary['autonomy_distribution'])
        print("Size distribution:", summary['size_distribution'])
    else:
        print("No robots to classify.") 