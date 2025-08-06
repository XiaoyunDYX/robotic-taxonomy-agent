import json
import os
import openai
from typing import Dict, Any

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

class GPTClassifier:
    def __init__(self, api_key: str, taxonomy_template: Dict[str, Any]):
        openai.api_key = api_key
        self.taxonomy_template = taxonomy_template

    def classify_robot(self, robot_info: Dict[str, Any], model: str = "gpt-4") -> Dict[str, Any]:
        prompt = self._build_prompt(robot_info)

        try:
            client = openai.OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a robotic classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
            classification = json.loads(content)
        except Exception as e:
            classification = {
                "error": str(e),
                "raw_response": content if 'content' in locals() else ""
            }

        # Ensure all required fields exist
        for field, template_val in self.taxonomy_template.items():
            if field not in classification:
                classification[field] = [] if isinstance(template_val, list) else "MISSING"

        return classification

    def _build_prompt(self, robot_info: Dict[str, Any]) -> str:
        info_text = json.dumps(robot_info, indent=2, ensure_ascii=False)
        # Force exact JSON schema with quoted keys, including "name"
        schema = json.dumps(self.taxonomy_template, indent=2, ensure_ascii=False)
        return f"""
You are an expert in robotic systems taxonomy. Given the following robot metadata:

{info_text}

Classify the robot according to the **8-level robotic taxonomy** below.
You **MUST** return **only** a valid JSON object with **exactly** these 10 fields (no extra keys, no explanation):

{schema}

- "name": string (the robot's name)
- "url": string (the robot's url)
- "domain": one of (Physical | Virtual | Hybrid)
- "kingdom": one of (Industrial | Medical | Service | Military | Agriculture | Space | Marine | Research | Entertainment)
- "morpho_motion_class": one of (Wheeled-Mobile | Tracked-Mobile | Legged-Humanoid | Legged-Animaloid | Flying-Drone | Swimming-Soft | Modular-Lattice | Swarm-Agent)
- "order": one of (Manual | Teleoperated | Autonomous | Semi-Autonomous | Collaborative | Swarm-Based)
- "sensing_family": one of (Vision-Based | LiDAR-Based | Tactile-Based | GPS-Based | Acoustic-Based | Chemical-Based | Multimodal | Minimal Sensing)
- "actuation_genus": one of (Electric | Hydraulic | Pneumatic | Smart Materials | Bio-Hybrid | Magnetic | Passive | Hybrid Actuation)
- "cognition_class": one of (None | Rule-Based | Model-Based AI | AI-Powered | Adaptive Learning | Reinforcement Learning | Generative AI)
- "application_species": list of one or more from [Surgery, Inspection, Transport, Assembly, Exploration, Surveillance, Companionship, Education, Mapping, Rescue, Entertainment, Agricultural Task, Construction, Maintenance, Environmental Monitoring]

Ensure **all** fields appear, even if guessed. Output **only** the JSON.  
""".strip()

if __name__ == "__main__":
    taxonomy_template = {
        "name": "Robot Name",
        "url": "https://example.com",
        "domain": "Physical",
        "kingdom": "Medical",
        "morpho_motion_class": "Legged-Humanoid",
        "order": "Autonomous",
        "sensing_family": "Vision-Based",
        "actuation_genus": "Electric",
        "cognition_class": "AI-Powered",
        "application_species": ["Surgery"]
    }

    # Load raw robot data
    with open("./data/robots_data.json", "r", encoding="utf-8") as f:
        robot_list = json.load(f)

    classifier = GPTClassifier(API_KEY, taxonomy_template)
    results = []
    for robot in robot_list:
        res = classifier.classify_robot(robot)
        if "error" in res:
            print(f"Error classifying robot '{robot.get('name', '')}': {res['error']}")
        results.append(res)

    os.makedirs("./data", exist_ok=True)
    with open("./data/classified_robots_gpt.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Classified {len(results)} robots and saved to ./data/classified_robots_gpt.json")

