#!/usr/bin/env python3
"""
Robot Taxonomy Agent - Main Application

This application scrapes the internet for robot examples, classifies them according to a taxonomy,
and displays the information graphically as a "tree of life" for robots.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_scraper.robot_scraper import RobotScraper
from classifier.robot_classifier import RobotClassifier
from visualizer.robot_tree_visualizer import RobotTreeVisualizer

class RobotTaxonomyAgent:
    def __init__(self):
        self.scraper = RobotScraper()
        self.classifier = RobotClassifier()
        self.visualizer = RobotTreeVisualizer()
        
    def run_full_pipeline(self, search_terms=None, launch_dashboard=True):
        """
        Run the complete pipeline: scrape -> classify -> visualize
        """
        print("🤖 Robot Taxonomy Agent Starting...")
        print("=" * 50)
        
        # Step 1: Web Scraping
        print("\n📡 Step 1: Scraping robot data from the internet...")
        if search_terms is None:
            search_terms = [
                "industrial robots",
                "service robots", 
                "humanoid robots",
                "medical robots",
                "military robots",
                "domestic robots",
                "educational robots",
                "space robots",
                "underwater robots",
                "aerial robots"
            ]
        
        robots_data = self.scraper.search_robots(search_terms)
        self.scraper.robots_data = robots_data
        self.scraper.save_data()
        print(f"✅ Found {len(robots_data)} robots")
        
        # Step 2: Classification
        print("\n🏷️  Step 2: Classifying robots according to taxonomy...")
        classified_robots = self.classifier.classify_robots(robots_data)
        
        # Perform clustering
        clustering_info = self.classifier.cluster_robots()
        self.classifier.save_classified_data()
        
        # Print classification summary
        summary = self.classifier.get_taxonomy_summary()
        print(f"✅ Classified {summary['total_robots']} robots")
        print("📊 Classification Summary:")
        print(f"   Domains: {summary['domain_distribution']}")
        print(f"   Kingdoms: {summary['kingdom_distribution']}")
        print(f"   Phyla: {summary['phylum_distribution']}")
        print(f"   Classes: {summary['class_distribution']}")
        print(f"   Orders: {summary['order_distribution']}")
        print(f"   Families: {summary['family_distribution']}")
        print(f"   Genera: {summary['genus_distribution']}")
        print(f"   Species: {summary['species_distribution']}")
        
        # Step 3: Visualization
        print("\n🌳 Step 3: Creating robot tree of life visualization...")
        self.visualizer.load_data()
        
        if launch_dashboard:
            print("\n🚀 Launching interactive dashboard...")
            print("📱 Open your browser to http://localhost:8050")
            app = self.visualizer.create_dashboard()
            app.run(debug=False, port=8050)
        else:
            # Create PNG visualizations
            self.visualizer.save_radial_tree_as_png()
            self.visualizer.save_phylogenetic_tree_as_png()
            self.visualizer.save_dendrogram_as_png()
            self.visualizer.save_cluster_as_png()
            self.visualizer.save_taxonomy_bar_charts_as_png()
            self.visualizer.save_simplified_tree_as_png()
            self.visualizer.save_taxonomy_summary_as_png()
            print("✅ PNG visualizations saved to data/")
        
        print("\n🎉 Robot Taxonomy Agent completed successfully!")
    
    def run_scraper_only(self, search_terms=None):
        """
        Run only the web scraper
        """
        print("📡 Running web scraper only...")
        if search_terms is None:
            search_terms = ["industrial robots", "service robots", "humanoid robots"]
        
        robots_data = self.scraper.search_robots(search_terms)
        self.scraper.robots_data = robots_data
        self.scraper.save_data()
        print(f"✅ Scraped {len(robots_data)} robots")
    
    def run_classifier_only(self):
        """
        Run only the classifier on existing data
        """
        print("🏷️  Running classifier only...")
        try:
            with open('./data/robots_data.json', 'r') as f:
                robots_data = json.load(f)
            
            classified_robots = self.classifier.classify_robots(robots_data)
            clustering_info = self.classifier.cluster_robots()
            self.classifier.save_classified_data()
            
            summary = self.classifier.get_taxonomy_summary()
            print(f"✅ Classified {summary['total_robots']} robots")
            print("📊 Classification Summary:")
            print(f"   Domains: {summary['domain_distribution']}")
            print(f"   Kingdoms: {summary['kingdom_distribution']}")
            print(f"   Phyla: {summary['phylum_distribution']}")
            print(f"   Classes: {summary['class_distribution']}")
            print(f"   Orders: {summary['order_distribution']}")
            print(f"   Families: {summary['family_distribution']}")
            print(f"   Genera: {summary['genus_distribution']}")
            print(f"   Species: {summary['species_distribution']}")
            
        except FileNotFoundError:
            print("❌ No robot data found. Please run the scraper first.")
    
    def run_visualizer_only(self, launch_dashboard=True):
        """
        Run only the visualizer on existing classified data
        """
        print("🌳 Running visualizer only...")
        try:
            self.visualizer.load_data()
            
            if launch_dashboard:
                print("🚀 Launching interactive dashboard...")
                print("📱 Open your browser to http://localhost:8050")
                app = self.visualizer.create_dashboard()
                app.run(debug=False, port=8050)
            else:
                self.visualizer.save_radial_tree_as_png()
                self.visualizer.save_phylogenetic_tree_as_png()
                self.visualizer.save_dendrogram_as_png()
                self.visualizer.save_cluster_as_png()
                self.visualizer.save_taxonomy_bar_charts_as_png()
                self.visualizer.save_simplified_tree_as_png()
                self.visualizer.save_taxonomy_summary_as_png()
                print("✅ PNG visualizations saved to data/")
                
        except FileNotFoundError:
            print("❌ No classified robot data found. Please run the classifier first.")

def main():
    parser = argparse.ArgumentParser(description="Robot Taxonomy Agent")
    parser.add_argument('--mode', choices=['full', 'scraper', 'classifier', 'visualizer'], 
                       default='full', help='Which components to run')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='Don\'t launch the interactive dashboard')
    parser.add_argument('--search-terms', nargs='+', 
                       help='Custom search terms for scraping')
    
    args = parser.parse_args()
    
    agent = RobotTaxonomyAgent()
    
    if args.mode == 'full':
        agent.run_full_pipeline(search_terms=args.search_terms, 
                              launch_dashboard=not args.no_dashboard)
    elif args.mode == 'scraper':
        agent.run_scraper_only(search_terms=args.search_terms)
    elif args.mode == 'classifier':
        agent.run_classifier_only()
    elif args.mode == 'visualizer':
        agent.run_visualizer_only(launch_dashboard=not args.no_dashboard)

if __name__ == "__main__":
    main() 