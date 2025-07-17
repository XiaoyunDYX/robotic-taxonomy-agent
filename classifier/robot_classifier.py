import json
import re
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class RobotTaxonomy:
    def __init__(self, taxonomy_file: str = "Robotic taxonomy.md"):
        """
        Initialize taxonomy from markdown file
        """
        self.taxonomy_file = taxonomy_file
        self.taxonomy = self._parse_taxonomy_from_markdown()
        self.keywords = self._extract_keywords()
        
    def _parse_taxonomy_from_markdown(self) -> Dict[str, Any]:
        """
        Parse the taxonomy structure from the markdown file
        """
        try:
            with open(self.taxonomy_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Warning: Taxonomy file {self.taxonomy_file} not found. Using default taxonomy.")
            return self._get_default_taxonomy()
        
        # Parse the hierarchical structure from markdown
        taxonomy = {
            "Domain": {},
            "Kingdom": {},
            "Phylum": {},
            "Class": {},
            "Order": {},
            "Family": {},
            "Genus": {},
            "Species": {}
        }
        
        # Extract Domain level
        domain_match = re.search(r'### Domain Level: Operational Environment\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if domain_match:
            domain_section = domain_match.group(1)
            domains = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', domain_section)
            for domain, description in domains:
                taxonomy["Domain"][domain.strip()] = description.strip()
        
        # Extract Kingdom level
        kingdom_match = re.search(r'### Kingdom Level: Application Domains\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if kingdom_match:
            kingdom_section = kingdom_match.group(1)
            kingdoms = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', kingdom_section)
            for kingdom, description in kingdoms:
                taxonomy["Kingdom"][kingdom.strip()] = description.strip()
        
        # Extract Phylum level
        phylum_match = re.search(r'### Phylum Level: Morphological Structure\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if phylum_match:
            phylum_section = phylum_match.group(1)
            phyla = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', phylum_section)
            for phylum, description in phyla:
                taxonomy["Phylum"][phylum.strip()] = description.strip()
        
        # Extract Class level
        class_match = re.search(r'### Class Level: Locomotion Mechanism\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if class_match:
            class_section = class_match.group(1)
            classes = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', class_section)
            for class_type, description in classes:
                taxonomy["Class"][class_type.strip()] = description.strip()
        
        # Extract Order level
        order_match = re.search(r'### Order Level: Autonomy and Control\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if order_match:
            order_section = order_match.group(1)
            orders = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', order_section)
            for order, description in orders:
                taxonomy["Order"][order.strip()] = description.strip()
        
        # Extract Family level
        family_match = re.search(r'### Family Level: Sensing Modalities\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if family_match:
            family_section = family_match.group(1)
            families = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', family_section)
            for family, description in families:
                taxonomy["Family"][family.strip()] = description.strip()
        
        # Extract Genus level
        genus_match = re.search(r'### Genus Level: Actuation Systems\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if genus_match:
            genus_section = genus_match.group(1)
            genera = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', genus_section)
            for genus, description in genera:
                taxonomy["Genus"][genus.strip()] = description.strip()
        
        # Extract Species level
        species_match = re.search(r'### Species Level: Application Specialization\s*\n\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if species_match:
            species_section = species_match.group(1)
            species_list = re.findall(r'\*\*(.*?)\*\*: (.*?)(?=\n|$)', species_section)
            for species, description in species_list:
                taxonomy["Species"][species.strip()] = description.strip()
        
        return taxonomy
    
    def _get_default_taxonomy(self) -> Dict[str, Any]:
        """
        Fallback taxonomy if markdown file is not available
        """
        return {
            "Domain": {
                "Physical": "Material world operation with direct environmental interaction",
                "Virtual": "Digital environment operation (software agents, simulated systems)",
                "Hybrid": "Bridging physical and virtual domains (AR/VR robotics, telepresence)"
            },
            "Kingdom": {
                "Industrial": "Manufacturing, production, factory automation",
                "Service": "Human assistance, domestic, commercial applications",
                "Medical": "Healthcare, surgical assistance, therapeutic applications",
                "Military": "Defense, security, tactical operations",
                "Research": "Scientific investigation and experimentation",
                "Entertainment": "Recreation, education, social interaction",
                "Agriculture": "Farming, crop management, agricultural automation",
                "Space": "Extraterrestrial exploration and operations",
                "Marine": "Underwater and surface water operations"
            },
            "Phylum": {
                "Manipulator": "Articulated arm systems with fixed bases",
                "Mobile": "Systems capable of translocation and navigation",
                "Humanoid": "Human-like bipedal morphology",
                "Modular": "Reconfigurable systems with interchangeable components",
                "Swarm": "Collective systems operating as coordinated groups",
                "Soft": "Compliant, deformable structures with flexible materials",
                "Hybrid_Morphology": "Combined rigid-soft or multi-modal body plans"
            },
            "Class": {
                "Static": "Fixed position systems",
                "Wheeled": "Wheel-based locomotion",
                "Legged": "Leg-based locomotion",
                "Flying": "Aerial locomotion",
                "Swimming": "Aquatic locomotion",
                "Morphing": "Shape-changing locomotion"
            },
            "Order": {
                "Teleoperated": "Human-controlled operation",
                "Semi_Autonomous": "Partial autonomous operation with human oversight",
                "Autonomous": "Fully autonomous operation",
                "Collaborative": "Human-robot collaborative operation"
            },
            "Family": {
                "Vision_Based": "Cameras and visual processing systems",
                "LiDAR_Based": "Laser-based distance and mapping systems",
                "Tactile_Based": "Touch and force sensing systems",
                "Multimodal": "Integration of multiple sensing technologies",
                "Minimal_Sensing": "Simple sensors with basic environmental awareness",
                "GPS_Navigation": "Satellite-based positioning systems",
                "Acoustic_Based": "Sound and ultrasonic sensing systems",
                "Chemical_Sensing": "Detection of chemical compounds and gases"
            },
            "Genus": {
                "Electric": "Electric motors and servo systems",
                "Hydraulic": "Fluid pressure-based actuation",
                "Pneumatic": "Compressed air actuation",
                "Hybrid_Actuation": "Combination of multiple actuation methods",
                "Smart_Materials": "Shape memory alloys, piezoelectric actuators",
                "Bio_Hybrid": "Integration of biological and artificial components",
                "Passive": "No active actuation, gravity or environmental forces",
                "Magnetic": ["magnetic", "magnet", "field", "magnetic"]
            },
            "Species": {
                "Assembly": "Manufacturing assembly line tasks",
                "Inspection": "Quality control and monitoring applications",
                "Transport": "Material handling and logistics operations",
                "Surgery": "Medical procedures requiring extreme precision",
                "Exploration": "Discovery and reconnaissance missions",
                "Maintenance": "Repair and upkeep operations",
                "Security": "Surveillance and protection tasks",
                "Education": "Teaching and learning assistance",
                "Companionship": "Social interaction and emotional support",
                "Agriculture_Specific": "Farming and crop management",
                "Rescue": "Emergency response and disaster relief",
                "Entertainment_Performance": "Shows, games, interactive experiences",
                "Mining": "Resource extraction and underground operations",
                "Construction": "Building and infrastructure development",
                "Environmental_Monitoring": "Ecosystem and pollution monitoring"
            }
        }
        
    def _extract_keywords(self) -> Dict[str, List[str]]:
        """
        Extract keywords from taxonomy for classification
        """
        keywords = {}
        
        for level, categories in self.taxonomy.items():
            for category, description in categories.items():
                # Extract keywords from description
                words = re.findall(r'\b\w+\b', description.lower())
                keywords[f"{level}.{category}"] = words
        
        return keywords
    
    def classify_robot(self, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a robot according to the hierarchical taxonomy
        """
        classification = {
            "Domain": {},
            "Kingdom": {},
            "Phylum": {},
            "Class": {},
            "Order": {},
            "Family": {},
            "Genus": {},
            "Species": {}
        }
        
        # Extract text for analysis
        text = self._extract_text(robot_data)
        text_lower = text.lower()
        
        # Classify by each taxonomic level
        classification["Domain"] = self._classify_domain(text_lower, robot_data)
        classification["Kingdom"] = self._classify_kingdom(text_lower, robot_data)
        classification["Phylum"] = self._classify_phylum(text_lower, robot_data)
        classification["Class"] = self._classify_class(text_lower, robot_data)
        classification["Order"] = self._classify_order(text_lower, robot_data)
        classification["Family"] = self._classify_family(text_lower, robot_data)
        classification["Genus"] = self._classify_genus(text_lower, robot_data)
        classification["Species"] = self._classify_species(text_lower, robot_data)
        
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
        
        # Keywords for each domain
        domain_keywords = {
            "Physical": ["physical", "real", "material", "hardware", "mechanical"],
            "Virtual": ["virtual", "digital", "software", "simulation", "computer"],
            "Hybrid": ["hybrid", "mixed", "augmented", "virtual reality", "ar", "vr"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                domain_scores[domain] = 0.8
        
        # Default to Physical if unclear
        if not domain_scores:
            domain_scores["Physical"] = 0.5
        
        return domain_scores
    
    def _classify_kingdom(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by kingdom
        """
        kingdom_scores = {}
        
        # Keywords for each kingdom
        kingdom_keywords = {
            "Industrial": ["industrial", "manufacturing", "factory", "production", "assembly"],
            "Service": ["service", "domestic", "home", "assistance", "cleaning"],
            "Medical": ["medical", "surgical", "healthcare", "hospital", "therapy"],
            "Military": ["military", "defense", "weapon", "combat", "surveillance"],
            "Research": ["research", "experimental", "laboratory", "exploration"],
            "Entertainment": ["entertainment", "game", "toy", "recreation", "performance"],
            "Agriculture": ["agriculture", "farming", "crop", "agricultural"],
            "Space": ["space", "satellite", "extraterrestrial", "orbital"],
            "Marine": ["marine", "underwater", "ocean", "submarine", "aquatic"]
        }
        
        for kingdom, keywords in kingdom_keywords.items():
            if any(keyword in text for keyword in keywords):
                kingdom_scores[kingdom] = 0.8
        
        # Default to Service if unclear
        if not kingdom_scores:
            kingdom_scores["Service"] = 0.5
        
        return kingdom_scores
    
    def _classify_phylum(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by phylum
        """
        phylum_scores = {}
        
        # Keywords for each phylum
        phylum_keywords = {
            "Manipulator": ["manipulator", "arm", "articulated", "fixed base"],
            "Mobile": ["mobile", "moving", "navigation", "locomotion"],
            "Humanoid": ["humanoid", "human-like", "bipedal", "anthropomorphic"],
            "Modular": ["modular", "reconfigurable", "interchangeable"],
            "Swarm": ["swarm", "collective", "group", "cooperative"],
            "Soft": ["soft", "compliant", "deformable", "flexible"],
            "Hybrid_Morphology": ["hybrid", "combined", "multi-modal"]
        }
        
        for phylum, keywords in phylum_keywords.items():
            if any(keyword in text for keyword in keywords):
                phylum_scores[phylum] = 0.8
        
        # Default to Mobile if unclear
        if not phylum_scores:
            phylum_scores["Mobile"] = 0.5
        
        return phylum_scores
    
    def _classify_class(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by class (locomotion mechanism)
        """
        class_scores = {}
        
        # Keywords for each class
        class_keywords = {
            "Static": ["static", "fixed", "stationary", "mounted"],
            "Wheeled": ["wheel", "wheeled", "drive", "track"],
            "Legged": ["leg", "legged", "biped", "quadruped", "walk"],
            "Flying": ["fly", "flying", "drone", "aerial", "helicopter"],
            "Swimming": ["swim", "swimming", "underwater", "aquatic"],
            "Morphing": ["morph", "shape-changing", "transform"]
        }
        
        for class_type, keywords in class_keywords.items():
            if any(keyword in text for keyword in keywords):
                class_scores[class_type] = 0.8
        
        # Default to Wheeled if unclear
        if not class_scores:
            class_scores["Wheeled"] = 0.5
        
        return class_scores
    
    def _classify_order(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by order (autonomy level)
        """
        order_scores = {}
        
        # Keywords for each order
        order_keywords = {
            "Teleoperated": ["teleoperated", "remote", "controlled", "manual"],
            "Semi_Autonomous": ["semi", "assisted", "supervised", "collaborative"],
            "Autonomous": ["autonomous", "ai", "artificial intelligence", "independent"],
            "Collaborative": ["collaborative", "cooperative", "human-robot"]
        }
        
        for order, keywords in order_keywords.items():
            if any(keyword in text for keyword in keywords):
                order_scores[order] = 0.8
        
        # Default to Semi_Autonomous if unclear
        if not order_scores:
            order_scores["Semi_Autonomous"] = 0.5
        
        return order_scores
    
    def _classify_family(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by family (sensing modality)
        """
        family_scores = {}
        
        # Keywords for each family
        family_keywords = {
            "Vision_Based": ["vision", "camera", "visual", "image"],
            "LiDAR_Based": ["lidar", "laser", "distance", "mapping"],
            "Tactile_Based": ["tactile", "touch", "force", "pressure"],
            "Multimodal": ["multimodal", "multiple", "integrated", "sensors"],
            "Minimal_Sensing": ["minimal", "simple", "basic", "sensor"],
            "GPS_Navigation": ["gps", "satellite", "positioning", "navigation"],
            "Acoustic_Based": ["acoustic", "sound", "ultrasonic", "audio"],
            "Chemical_Sensing": ["chemical", "gas", "compound", "detection"]
        }
        
        for family, keywords in family_keywords.items():
            if any(keyword in text for keyword in keywords):
                family_scores[family] = 0.8
        
        # Default to Vision_Based if unclear
        if not family_scores:
            family_scores["Vision_Based"] = 0.5
        
        return family_scores
    
    def _classify_genus(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by genus (actuation system)
        """
        genus_scores = {}
        
        # Keywords for each genus
        genus_keywords = {
            "Electric": ["electric", "motor", "servo", "electronic"],
            "Hydraulic": ["hydraulic", "fluid", "pressure", "oil"],
            "Pneumatic": ["pneumatic", "air", "compressed", "pneumatic"],
            "Hybrid_Actuation": ["hybrid", "combined", "multiple", "actuation"],
            "Smart_Materials": ["smart", "memory", "piezoelectric", "alloy"],
            "Bio_Hybrid": ["bio", "biological", "hybrid", "living"],
            "Passive": ["passive", "gravity", "environmental", "no actuation"],
            "Magnetic": "Magnetic field-based actuation"
        }
        
        for genus, keywords in genus_keywords.items():
            if any(keyword in text for keyword in keywords):
                genus_scores[genus] = 0.8
        
        # Default to Electric if unclear
        if not genus_scores:
            genus_scores["Electric"] = 0.5
        
        return genus_scores
    
    def _classify_species(self, text: str, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify robot by species (application specialization)
        """
        species_scores = {}
        
        # Keywords for each species
        species_keywords = {
            "Assembly": ["assembly", "manufacturing", "production", "line"],
            "Inspection": ["inspection", "quality", "control", "monitoring"],
            "Transport": ["transport", "material", "handling", "logistics"],
            "Surgery": ["surgery", "surgical", "medical", "procedure"],
            "Exploration": ["exploration", "discovery", "reconnaissance", "mission"],
            "Maintenance": ["maintenance", "repair", "upkeep", "service"],
            "Security": ["security", "surveillance", "protection", "guard"],
            "Education": ["education", "teaching", "learning", "training"],
            "Companionship": ["companion", "social", "interaction", "emotional"],
            "Agriculture_Specific": ["agriculture", "farming", "crop", "agricultural"],
            "Rescue": ["rescue", "emergency", "disaster", "relief"],
            "Entertainment_Performance": ["entertainment", "performance", "show", "game"],
            "Mining": ["mining", "extraction", "underground", "resource"],
            "Construction": ["construction", "building", "infrastructure", "development"],
            "Environmental_Monitoring": ["environmental", "ecosystem", "pollution", "monitoring"]
        }
        
        for species, keywords in species_keywords.items():
            if any(keyword in text for keyword in keywords):
                species_scores[species] = 0.8
        
        # Default to Exploration if unclear
        if not species_scores:
            species_scores["Exploration"] = 0.5
        
        return species_scores

class RobotClassifier:
    def __init__(self, taxonomy_file: str = "Robotic taxonomy.md"):
        self.taxonomy = RobotTaxonomy(taxonomy_file)
        self.classified_robots = []
        
    def classify_robots(self, robots_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a list of robots according to the hierarchical taxonomy
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
            'kingdom_distribution': {},
            'phylum_distribution': {},
            'class_distribution': {},
            'order_distribution': {},
            'family_distribution': {},
            'genus_distribution': {},
            'species_distribution': {}
        }
        
        for robot in self.classified_robots:
            classification = robot.get('classification', {})
            
            # Count classifications for each taxonomic level
            for level in ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
                level_distribution = summary[f'{level.lower()}_distribution']
                for category, score in classification.get(level, {}).items():
                    if score > 0.5:  # Only count confident classifications
                        level_distribution[category] = level_distribution.get(category, 0) + 1
        
        return summary
    
    def get_taxonomy_structure(self) -> Dict[str, Any]:
        """
        Get the complete taxonomy structure
        """
        return self.taxonomy.taxonomy

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
        print("Kingdom distribution:", summary['kingdom_distribution'])
        print("Phylum distribution:", summary['phylum_distribution'])
        print("Class distribution:", summary['class_distribution'])
        print("Order distribution:", summary['order_distribution'])
        print("Family distribution:", summary['family_distribution'])
        print("Genus distribution:", summary['genus_distribution'])
        print("Species distribution:", summary['species_distribution'])
        
        # Print taxonomy structure
        print("\nTaxonomy structure:")
        taxonomy_structure = classifier.get_taxonomy_structure()
        for level, categories in taxonomy_structure.items():
            print(f"{level}: {list(categories.keys())}")
    else:
        print("No robots to classify.") 