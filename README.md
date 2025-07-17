# Robot Taxonomy Agent 🤖

A comprehensive system that scours the internet for robot examples, classifies them according to a hierarchical taxonomy, and displays the information graphically as a "tree of life" for robots.

## Features

- **🌐 Web Scraping**: Automatically searches Wikipedia, manufacturer websites, and robot databases
- **🏷️ Intelligent Classification**: Categorizes robots by Domain, Mobility, Autonomy, and Size
- **🌳 Interactive Visualization**: Creates a phylogenetic-style tree showing robot relationships
- **📊 Dashboard**: Interactive web interface for exploring robot taxonomy
- **🤖 Machine Learning**: Uses clustering to discover natural robot groupings

## Taxonomy Structure

The system classifies robots according to four main dimensions:

### Domain
- **Industrial**: Manufacturing, Logistics, Construction
- **Service**: Healthcare, Domestic, Commercial
- **Research**: Exploration, Laboratory, Educational
- **Military**: Combat, Support, Surveillance

### Mobility
- **Stationary**: Fixed, Mounted, Permanent
- **Mobile**: 
  - Wheeled: Differential Drive, Omnidirectional, Tracked
  - Legged: Bipedal, Quadrupedal, Hexapod
  - Flying: Fixed-wing, Rotary-wing, Multi-rotor
  - Swimming: Underwater, Surface, Hybrid

### Autonomy
- **Teleoperated**: Remote Control, Human-in-the-loop
- **Semi-autonomous**: Supervised, Assisted
- **Autonomous**: Fully Autonomous, AI-driven, Self-learning

### Size
- **Nano**: Microscopic, Molecular
- **Micro**: Millimeter, Centimeter
- **Small**: Handheld, Portable
- **Medium**: Human-sized, Table-top
- **Large**: Industrial, Vehicle-sized
- **Macro**: Building-sized, Infrastructure

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd robot-taxonomy-agent
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python src/main.py --help
```

## Usage

### Full Pipeline (Recommended)

Run the complete system - scraping, classification, and visualization:

```bash
python src/main.py --mode full
```

This will:
1. Scrape robot data from the internet
2. Classify robots according to the taxonomy
3. Launch an interactive dashboard at http://localhost:8050

### Individual Components

#### Web Scraping Only
```bash
python src/main.py --mode scraper
```

#### Classification Only (requires scraped data)
```bash
python src/main.py --mode classifier
```

#### Visualization Only (requires classified data)
```bash
python src/main.py --mode visualizer
```

### Custom Search Terms

Specify custom search terms for scraping:

```bash
python src/main.py --mode scraper --search-terms "space robots" "underwater robots" "medical robots"
```

### Static Visualizations

Generate static HTML visualizations without launching the dashboard:

```bash
python src/main.py --mode full --no-dashboard
```

## Project Structure

```
robot-taxonomy-agent/
├── src/
│   └── main.py                 # Main application orchestrator
├── web_scraper/
│   └── robot_scraper.py        # Web scraping functionality
├── classifier/
│   └── robot_classifier.py     # Taxonomy classification
├── visualizer/
│   └── robot_tree_visualizer.py # Visualization and dashboard
├── data/
│   ├── robots_data.json        # Raw scraped data
│   └── classified_robots.json  # Classified robot data
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Dashboard Features

The interactive dashboard provides:

- **Tree Visualization**: Hierarchical view of robot relationships
- **Filtering**: Filter robots by domain, mobility, autonomy
- **Statistics**: Summary of classification results
- **Robot Details**: Click on robots to see detailed information
- **Cluster Analysis**: Machine learning-based robot groupings

## Data Sources

The system scrapes data from:

- **Wikipedia**: Robot categories and articles
- **Manufacturer Websites**: ABB, FANUC, KUKA, Yaskawa
- **Robot Databases**: Various robot information sources

## Customization

### Adding New Robot Categories

Edit `classifier/robot_classifier.py` to modify the taxonomy structure:

```python
self.taxonomy = {
    "Domain": {
        "Your New Domain": {
            "Subcategory": ["Specific Types"]
        }
    }
    # ... rest of taxonomy
}
```

### Custom Search Sources

Modify `web_scraper/robot_scraper.py` to add new data sources:

```python
def _search_custom_source(self, search_term: str):
    # Add your custom scraping logic here
    pass
```

### Visualization Customization

Edit `visualizer/robot_tree_visualizer.py` to modify the visual representation:

```python
def create_custom_visualization(self):
    # Add your custom visualization logic here
    pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **No Data Found**: Run the scraper first
   ```bash
   python src/main.py --mode scraper
   ```

3. **Dashboard Not Loading**: Check if port 8050 is available
   ```bash
   python src/main.py --mode visualizer --no-dashboard
   ```

4. **Memory Issues**: Reduce the number of search terms or robots processed

### Performance Tips

- Use specific search terms to reduce scraping time
- Run components separately for debugging
- Use `--no-dashboard` for faster processing
- Limit the number of robots processed for testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by biological taxonomy and phylogenetic trees
- Uses data from Wikipedia and various robot manufacturers
- Built with Python, Dash, Plotly, and NetworkX

---

**Happy Robot Exploring! 🤖🌳** 

## Code Explanation

This is a **Robot Taxonomy Agent** - a comprehensive system that automatically discovers, classifies, and visualizes robots from the internet. Here's how it works:

### 🏗️ **Architecture Overview**

The system has three main components:

1. **🌐 Web Scraper** (`web_scraper/robot_scraper.py`)
   - Searches Wikipedia and manufacturer websites for robot information
   - Extracts robot names, descriptions, specifications, and metadata
   - Saves raw data to JSON files

2. **🏷️ Classifier** (`classifier/robot_classifier.py`)
   - Uses a hierarchical taxonomy to categorize robots
   - Classifies by: Domain, Mobility, Autonomy, and Size
   - Applies machine learning clustering to find natural groupings
   - Uses keyword matching and TF-IDF analysis

3. **🌳 Visualizer** (`visualizer/robot_tree_visualizer.py`)
   - Creates interactive "tree of life" style visualizations
   - Builds a Dash web dashboard for exploration
   - Generates phylogenetic-style trees showing robot relationships

### 🔄 **Pipeline Flow**

```
Internet → Scraper → Raw Data → Classifier → Classified Data → Visualizer → Dashboard
```

### 📊 **Taxonomy Structure**

The system uses a sophisticated 4-dimensional classification:

- **Domain**: Industrial, Service, Research, Military
- **Mobility**: Stationary, Mobile (Wheeled, Legged, Flying, Swimming)
- **Autonomy**: Teleoperated, Semi-autonomous, Autonomous
- **Size**: Nano, Micro, Small, Medium, Large, Macro

##  How to Run It

### **Step 1: Install Dependencies** 