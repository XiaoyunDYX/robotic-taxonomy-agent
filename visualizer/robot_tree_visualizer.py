import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from typing import Dict, List, Any, Optional
import pandas as pd
import kaleido  # For PNG export
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict

class RobotTreeVisualizer:
    def __init__(self):
        self.graph = nx.Graph()
        self.robots_data = []
        self.taxonomy_hierarchy = {}
        
    def load_data(self, filename: str = 'classified_robots.json'):
        """
        Load classified robot data
        """
        try:
            with open(f'./data/{filename}', 'r') as f:
                self.robots_data = json.load(f)
            print(f"Loaded {len(self.robots_data)} robots")
        except FileNotFoundError:
            print(f"File {filename} not found")
            self.robots_data = []
    
    def build_taxonomy_tree(self) -> nx.Graph:
        """
        Build a hierarchical tree structure based on the new taxonomy
        """
        # Define the new hierarchical taxonomy structure
        taxonomy = {
            "Robots": {
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
                    "Magnetic": "Magnetic field-based actuation"
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
        }
        
        self.taxonomy_hierarchy = taxonomy
        self.graph = nx.Graph()
        
        # Build the tree structure
        self._add_taxonomy_nodes(taxonomy, parent="")
        
        # Add robot nodes and connect them to taxonomy
        self._add_robot_nodes()
        
        return self.graph
    
    def _add_taxonomy_nodes(self, taxonomy: Dict, parent: str):
        """
        Recursively add taxonomy nodes to the graph
        """
        for key, value in taxonomy.items():
            node_id = f"{parent}.{key}" if parent else key
            self.graph.add_node(node_id, type="taxonomy", level=len(node_id.split('.'))-1)
            
            if parent:
                self.graph.add_edge(parent, node_id)
            
            if isinstance(value, dict):
                self._add_taxonomy_nodes(value, node_id)
            elif isinstance(value, str):
                # For the new structure, values are descriptions
                pass
    
    def _add_robot_nodes(self):
        """
        Add robot nodes and connect them to appropriate taxonomy nodes
        """
        for robot in self.robots_data:
            robot_id = f"robot_{robot['name'].replace(' ', '_')}"
            self.graph.add_node(robot_id, 
                              type="robot", 
                              name=robot['name'],
                              description=robot.get('description', ''),
                              manufacturer=robot.get('manufacturer', ''),
                              level=10)  # Robots are at the bottom level
            
            # Connect robot to taxonomy based on classification
            classification = robot.get('classification', {})
            
            # Connect to each taxonomic level
            taxonomic_levels = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
            
            for level in taxonomic_levels:
                for category, score in classification.get(level, {}).items():
                    if score > 0.5:
                        level_node = f"Robots.{level}.{category}"
                        if level_node in self.graph:
                            self.graph.add_edge(level_node, robot_id, weight=score)
    
    def create_radial_tree_of_life(self) -> go.Figure:
        """
        Create a radial tree of life visualization similar to biological taxonomy
        """
        if not self.graph.nodes():
            self.build_taxonomy_tree()
        
        # Use radial layout for tree of life effect
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Convert to radial coordinates
        radial_pos = {}
        for node, (x, y) in pos.items():
            # Convert to polar coordinates
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            radial_pos[node] = (r, theta)
        
        # Separate nodes by type and level
        taxonomy_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'taxonomy']
        robot_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'robot']
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges with curved lines for tree effect
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Create curved edges
            t = np.linspace(0, 1, 50)
            x_curve = x0 + (x1 - x0) * t
            y_curve = y0 + (y1 - y0) * t
            
            edge_x.extend(x_curve)
            edge_y.extend(y_curve)
            edge_colors.extend(['#888'] * len(t))
            
            edge_x.append(None)
            edge_y.append(None)
            edge_colors.append(None)
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add taxonomy nodes by level with different colors
        level_colors = {
            0: '#1f77b4',  # Root
            1: '#ff7f0e',  # Domain
            2: '#2ca02c',  # Kingdom
            3: '#d62728',  # Phylum
            4: '#9467bd',  # Class
            5: '#8c564b',  # Order
            6: '#e377c2',  # Family
            7: '#7f7f7f',  # Genus
            8: '#bcbd22'   # Species
        }
        
        for level in range(9):
            level_nodes = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'taxonomy' and d.get('level') == level]
            
            if level_nodes:
                x_coords = [pos[node][0] for node in level_nodes]
                y_coords = [pos[node][1] for node in level_nodes]
                node_names = [node.split('.')[-1] for node in level_nodes]
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=15 + level * 2,
                        color=level_colors.get(level, '#1f77b4'),
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    text=node_names,
                    textposition="middle center",
                    textfont=dict(size=8 + level),
                    name=f'Level {level}',
                    hovertemplate='<b>%{text}</b><br>Level: %{customdata}<extra></extra>',
                    customdata=[level] * len(level_nodes),
                    showlegend=(level < 3)  # Only show first 3 levels in legend
                ))
        
        # Add robot nodes as leaves
        if robot_nodes:
            robot_x = [pos[node][0] for node in robot_nodes]
            robot_y = [pos[node][1] for node in robot_nodes]
            robot_names = [self.graph.nodes[node]['name'] for node in robot_nodes]
            
            fig.add_trace(go.Scatter(
                x=robot_x, y=robot_y,
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    symbol='circle',
                    opacity=0.7
                ),
                name='Robots',
                hovertemplate='<b>%{text}</b><extra></extra>',
                text=robot_names,
                showlegend=True
            ))
        
        fig.update_layout(
            title="Tree of Robotic Life - Radial View",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_phylogenetic_tree(self) -> go.Figure:
        """
        Create a phylogenetic-style tree visualization
        """
        if not self.graph.nodes():
            self.build_taxonomy_tree()
        
        # Create hierarchical layout
        pos = nx.spring_layout(self.graph, k=3, iterations=100)
        
        # Separate nodes by type
        taxonomy_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'taxonomy']
        robot_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'robot']
        
        fig = go.Figure()
        
        # Add edges with phylogenetic style
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Create straight phylogenetic-style edges
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#333'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add taxonomy nodes with phylogenetic colors
        level_colors = {
            0: '#1f77b4',  # Root
            1: '#ff7f0e',  # Domain
            2: '#2ca02c',  # Kingdom
            3: '#d62728',  # Phylum
            4: '#9467bd',  # Class
            5: '#8c564b',  # Order
            6: '#e377c2',  # Family
            7: '#7f7f7f',  # Genus
            8: '#bcbd22'   # Species
        }
        
        for level in range(9):
            level_nodes = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'taxonomy' and d.get('level') == level]
            
            if level_nodes:
                x_coords = [pos[node][0] for node in level_nodes]
                y_coords = [pos[node][1] for node in level_nodes]
                node_names = [node.split('.')[-1] for node in level_nodes]
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=20 + level * 3,
                        color=level_colors.get(level, '#1f77b4'),
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    text=node_names,
                    textposition="middle center",
                    textfont=dict(size=10 + level),
                    name=f'Taxonomic Level {level}',
                    hovertemplate='<b>%{text}</b><br>Level: %{customdata}<extra></extra>',
                    customdata=[level] * len(level_nodes)
                ))
        
        # Add robot nodes as terminal taxa
        if robot_nodes:
            robot_x = [pos[node][0] for node in robot_nodes]
            robot_y = [pos[node][1] for node in robot_nodes]
            robot_names = [self.graph.nodes[node]['name'] for node in robot_nodes]
            
            fig.add_trace(go.Scatter(
                x=robot_x, y=robot_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='circle',
                    opacity=0.8
                ),
                name='Robot Species',
                hovertemplate='<b>%{text}</b><extra></extra>',
                text=robot_names
            ))
        
        fig.update_layout(
            title="Tree of Robotic Life - Phylogenetic View",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=1000
        )
        
        return fig
    
    def create_hierarchical_dendrogram(self) -> go.Figure:
        """
        Create a hierarchical dendrogram visualization
        """
        if not self.graph.nodes():
            self.build_taxonomy_tree()
        
        # Use hierarchical layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Separate nodes by type
        taxonomy_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'taxonomy']
        robot_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'robot']
        
        fig = go.Figure()
        
        # Add edges with dendrogram style
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Create dendrogram-style edges
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#666'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add taxonomy nodes with hierarchical colors
        level_colors = {
            0: '#1f77b4',  # Root
            1: '#ff7f0e',  # Domain
            2: '#2ca02c',  # Kingdom
            3: '#d62728',  # Phylum
            4: '#9467bd',  # Class
            5: '#8c564b',  # Order
            6: '#e377c2',  # Family
            7: '#7f7f7f',  # Genus
            8: '#bcbd22'   # Species
        }
        
        for level in range(9):
            level_nodes = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'taxonomy' and d.get('level') == level]
            
            if level_nodes:
                x_coords = [pos[node][0] for node in level_nodes]
                y_coords = [pos[node][1] for node in level_nodes]
                node_names = [node.split('.')[-1] for node in level_nodes]
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=18 + level * 2,
                        color=level_colors.get(level, '#1f77b4'),
                        symbol='diamond' if level < 3 else 'circle',
                        line=dict(width=2, color='white')
                    ),
                    text=node_names,
                    textposition="middle center",
                    textfont=dict(size=9 + level),
                    name=f'Level {level}',
                    hovertemplate='<b>%{text}</b><br>Hierarchical Level: %{customdata}<extra></extra>',
                    customdata=[level] * len(level_nodes)
                ))
        
        # Add robot nodes as leaves
        if robot_nodes:
            robot_x = [pos[node][0] for node in robot_nodes]
            robot_y = [pos[node][1] for node in robot_nodes]
            robot_names = [self.graph.nodes[node]['name'] for node in robot_nodes]
            
            fig.add_trace(go.Scatter(
                x=robot_x, y=robot_y,
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    symbol='circle',
                    opacity=0.7
                ),
                name='Robot Leaves',
                hovertemplate='<b>%{text}</b><extra></extra>',
                text=robot_names
            ))
        
        fig.update_layout(
            title="Tree of Robotic Life - Hierarchical Dendrogram",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=1000
        )
        
        return fig

    def create_taxonomy_bar_charts(self) -> go.Figure:
        """
        Create bar charts showing the distribution of robots across taxonomic levels
        """
        if not self.robots_data:
            return go.Figure()
        
        # Define taxonomic levels
        taxonomic_levels = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        
        # Create subplots for each taxonomic level
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=taxonomic_levels,
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Define colors for each level
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, level in enumerate(taxonomic_levels):
            # Count robots in each category for this level
            category_counts = defaultdict(int)
            
            for robot in self.robots_data:
                classification = robot.get('classification', {})
                if level in classification:
                    level_data = classification[level]
                    if isinstance(level_data, dict):
                        # Find the category with the highest score
                        if level_data:
                            best_category = max(level_data.items(), key=lambda x: x[1])
                            category_counts[best_category[0]] += 1
            
            # Sort categories by count
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            categories = [cat for cat, count in sorted_categories]
            counts = [count for cat, count in sorted_categories]
            
            # Calculate row and column for subplot
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=counts,
                    name=level,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Update layout for this subplot
            fig.update_xaxes(title_text="Categories", row=row, col=col, tickangle=45)
            fig.update_yaxes(title_text="Count", row=row, col=col)
        
        fig.update_layout(
            title="Robot Distribution Across Taxonomic Levels",
            height=1200,
            width=1400,
            showlegend=False
        )
        
        return fig
    
    def create_simplified_tree(self) -> go.Figure:
        """
        Create a simplified tree showing only the taxonomic structure without individual robots
        """
        # Define the taxonomic hierarchy structure
        taxonomy_structure = {
            "Robots": {
                "Physical": {
                    "Industrial": {
                        "Manipulator": {
                            "Static": {
                                "Teleoperated": {
                                    "Vision_Based": {
                                        "Electric": {
                                            "Assembly": {},
                                            "Inspection": {},
                                            "Transport": {}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "Service": {
                        "Mobile": {
                            "Wheeled": {
                                "Semi_Autonomous": {
                                    "LiDAR_Based": {
                                        "Electric": {
                                            "Transport": {},
                                            "Security": {}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "Medical": {
                        "Manipulator": {
                            "Static": {
                                "Teleoperated": {
                                    "Vision_Based": {
                                        "Electric": {
                                            "Surgery": {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "Virtual": {
                    "Research": {
                        "Modular": {
                            "Static": {
                                "Autonomous": {
                                    "Multimodal": {
                                        "Electric": {
                                            "Education": {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "Hybrid": {
                    "Space": {
                        "Mobile": {
                            "Wheeled": {
                                "Autonomous": {
                                    "Vision_Based": {
                                        "Electric": {
                                            "Exploration": {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Create hierarchical layout
        fig = go.Figure()
        
        # Define node positions manually for better control
        node_positions = {
            "Robots": (0, 0),
            "Physical": (-2, -1), "Virtual": (0, -1), "Hybrid": (2, -1),
            "Industrial": (-3, -2), "Service": (-1, -2), "Medical": (1, -2),
            "Research": (0, -2), "Space": (2, -2),
            "Manipulator": (-3.5, -3), "Mobile": (-1.5, -3), "Modular": (0.5, -3),
            "Static": (-3.5, -4), "Wheeled": (-1.5, -4),
            "Teleoperated": (-3.5, -5), "Semi_Autonomous": (-1.5, -5), "Autonomous": (0.5, -5),
            "Vision_Based": (-3.5, -6), "LiDAR_Based": (-1.5, -6), "Multimodal": (0.5, -6),
            "Electric": (-3.5, -7), "Electric2": (-1.5, -7), "Electric3": (0.5, -7),
            "Assembly": (-4, -8), "Inspection": (-3, -8), "Transport": (-2, -8),
            "Transport2": (-1, -8), "Security": (0, -8), "Surgery": (1, -8),
            "Education": (0.5, -8), "Exploration": (2.5, -8)
        }
        
        # Add edges
        edges = [
            ("Robots", "Physical"), ("Robots", "Virtual"), ("Robots", "Hybrid"),
            ("Physical", "Industrial"), ("Physical", "Service"), ("Physical", "Medical"),
            ("Virtual", "Research"), ("Hybrid", "Space"),
            ("Industrial", "Manipulator"), ("Service", "Mobile"), ("Research", "Modular"),
            ("Manipulator", "Static"), ("Mobile", "Wheeled"),
            ("Static", "Teleoperated"), ("Wheeled", "Semi_Autonomous"), ("Modular", "Autonomous"),
            ("Teleoperated", "Vision_Based"), ("Semi_Autonomous", "LiDAR_Based"), ("Autonomous", "Multimodal"),
            ("Vision_Based", "Electric"), ("LiDAR_Based", "Electric2"), ("Multimodal", "Electric3"),
            ("Electric", "Assembly"), ("Electric", "Inspection"), ("Electric", "Transport"),
            ("Electric2", "Transport2"), ("Electric2", "Security"), ("Electric", "Surgery"),
            ("Electric3", "Education"), ("Electric", "Exploration")
        ]
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in edges:
            if edge[0] in node_positions and edge[1] in node_positions:
                x0, y0 = node_positions[edge[0]]
                x1, y1 = node_positions[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#333'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes with different colors by level
        level_colors = {
            0: '#1f77b4',  # Root
            1: '#ff7f0e',  # Domain
            2: '#2ca02c',  # Kingdom
            3: '#d62728',  # Phylum
            4: '#9467bd',  # Class
            5: '#8c564b',  # Order
            6: '#e377c2',  # Family
            7: '#7f7f7f',  # Genus
            8: '#bcbd22'   # Species
        }
        
        for node, (x, y) in node_positions.items():
            # Determine level based on node name
            level = 0
            if node in ["Physical", "Virtual", "Hybrid"]:
                level = 1
            elif node in ["Industrial", "Service", "Medical", "Research", "Space"]:
                level = 2
            elif node in ["Manipulator", "Mobile", "Modular"]:
                level = 3
            elif node in ["Static", "Wheeled"]:
                level = 4
            elif node in ["Teleoperated", "Semi_Autonomous", "Autonomous"]:
                level = 5
            elif node in ["Vision_Based", "LiDAR_Based", "Multimodal"]:
                level = 6
            elif node.startswith("Electric"):
                level = 7
            else:
                level = 8
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=20 + level * 3,
                    color=level_colors.get(level, '#1f77b4'),
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                text=[node],
                textposition="middle center",
                textfont=dict(size=10 + level),
                name=f'Level {level}',
                showlegend=False,
                hovertemplate=f'<b>{node}</b><br>Level: {level}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Simplified Tree of Robotic Life - Taxonomic Structure",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=1000
        )
        
        return fig
    
    def create_taxonomy_summary_chart(self) -> go.Figure:
        """
        Create a summary chart showing the top categories at each taxonomic level
        """
        if not self.robots_data:
            return go.Figure()
        
        # Define taxonomic levels
        taxonomic_levels = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        
        # Get top 5 categories for each level
        top_categories = {}
        
        for level in taxonomic_levels:
            category_counts = defaultdict(int)
            
            for robot in self.robots_data:
                classification = robot.get('classification', {})
                if level in classification:
                    level_data = classification[level]
                    if isinstance(level_data, dict):
                        if level_data:
                            best_category = max(level_data.items(), key=lambda x: x[1])
                            category_counts[best_category[0]] += 1
            
            # Get top 5 categories
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            top_categories[level] = sorted_categories[:5]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, level in enumerate(taxonomic_levels):
            if level in top_categories:
                categories = [cat for cat, count in top_categories[level]]
                counts = [count for cat, count in top_categories[level]]
                
                fig.add_trace(go.Bar(
                    y=[f"{level}: {cat}" for cat in categories],
                    x=counts,
                    orientation='h',
                    name=level,
                    marker_color=colors[i],
                    showlegend=False
                ))
        
        fig.update_layout(
            title="Top Categories at Each Taxonomic Level",
            xaxis_title="Number of Robots",
            yaxis_title="Taxonomic Categories",
            height=800,
            width=1200,
            showlegend=False
        )
        
        return fig

    def save_radial_tree_as_png(self, filename: str = 'robot_radial_tree.png'):
        """
        Save the radial tree visualization as a PNG file
        """
        fig = self.create_radial_tree_of_life()
        fig.write_image(f"./data/{filename}", width=1400, height=1000)
        print(f"✅ Radial tree visualization saved as PNG: data/{filename}")
    
    def save_phylogenetic_tree_as_png(self, filename: str = 'robot_phylogenetic_tree.png'):
        """
        Save the phylogenetic tree visualization as a PNG file
        """
        fig = self.create_phylogenetic_tree()
        fig.write_image(f"./data/{filename}", width=1400, height=1000)
        print(f"✅ Phylogenetic tree visualization saved as PNG: data/{filename}")
    
    def save_dendrogram_as_png(self, filename: str = 'robot_dendrogram.png'):
        """
        Save the hierarchical dendrogram visualization as a PNG file
        """
        fig = self.create_hierarchical_dendrogram()
        fig.write_image(f"./data/{filename}", width=1400, height=1000)
        print(f"✅ Dendrogram visualization saved as PNG: data/{filename}")
    
    def create_cluster_visualization(self) -> go.Figure:
        """
        Create a cluster visualization based on robot similarities
        """
        if not self.robots_data:
            return go.Figure()
        
        # Extract features for clustering visualization
        features = []
        robot_names = []
        
        for robot in self.robots_data:
            robot_names.append(robot['name'])
            
            # Create feature vector
            feature_vector = []
            
            # Domain features
            domain_scores = robot.get('classification', {}).get('Domain', {})
            feature_vector.extend([
                domain_scores.get('Physical', 0),
                domain_scores.get('Virtual', 0),
                domain_scores.get('Hybrid', 0)
            ])
            
            # Kingdom features
            kingdom_scores = robot.get('classification', {}).get('Kingdom', {})
            feature_vector.extend([
                kingdom_scores.get('Industrial', 0),
                kingdom_scores.get('Service', 0),
                kingdom_scores.get('Medical', 0),
                kingdom_scores.get('Military', 0),
                kingdom_scores.get('Research', 0),
                kingdom_scores.get('Entertainment', 0),
                kingdom_scores.get('Agriculture', 0),
                kingdom_scores.get('Space', 0),
                kingdom_scores.get('Marine', 0)
            ])
            
            # Phylum features
            phylum_scores = robot.get('classification', {}).get('Phylum', {})
            feature_vector.extend([
                phylum_scores.get('Manipulator', 0),
                phylum_scores.get('Mobile', 0),
                phylum_scores.get('Humanoid', 0),
                phylum_scores.get('Modular', 0),
                phylum_scores.get('Swarm', 0),
                phylum_scores.get('Soft', 0),
                phylum_scores.get('Hybrid_Morphology', 0)
            ])
            
            # Class features
            class_scores = robot.get('classification', {}).get('Class', {})
            feature_vector.extend([
                class_scores.get('Static', 0),
                class_scores.get('Wheeled', 0),
                class_scores.get('Legged', 0),
                class_scores.get('Flying', 0),
                class_scores.get('Swimming', 0),
                class_scores.get('Morphing', 0)
            ])
            
            # Order features
            order_scores = robot.get('classification', {}).get('Order', {})
            feature_vector.extend([
                order_scores.get('Teleoperated', 0),
                order_scores.get('Semi_Autonomous', 0),
                order_scores.get('Autonomous', 0),
                order_scores.get('Collaborative', 0)
            ])
            
            # Family features
            family_scores = robot.get('classification', {}).get('Family', {})
            feature_vector.extend([
                family_scores.get('Vision_Based', 0),
                family_scores.get('LiDAR_Based', 0),
                family_scores.get('Tactile_Based', 0),
                family_scores.get('Multimodal', 0),
                family_scores.get('Minimal_Sensing', 0),
                family_scores.get('GPS_Navigation', 0),
                family_scores.get('Acoustic_Based', 0),
                family_scores.get('Chemical_Sensing', 0)
            ])
            
            # Genus features
            genus_scores = robot.get('classification', {}).get('Genus', {})
            feature_vector.extend([
                genus_scores.get('Electric', 0),
                genus_scores.get('Hydraulic', 0),
                genus_scores.get('Pneumatic', 0),
                genus_scores.get('Hybrid_Actuation', 0),
                genus_scores.get('Smart_Materials', 0),
                genus_scores.get('Bio_Hybrid', 0),
                genus_scores.get('Passive', 0),
                genus_scores.get('Magnetic', 0)
            ])
            
            # Species features
            species_scores = robot.get('classification', {}).get('Species', {})
            feature_vector.extend([
                species_scores.get('Assembly', 0),
                species_scores.get('Inspection', 0),
                species_scores.get('Transport', 0),
                species_scores.get('Surgery', 0),
                species_scores.get('Exploration', 0),
                species_scores.get('Maintenance', 0),
                species_scores.get('Security', 0),
                species_scores.get('Education', 0),
                species_scores.get('Companionship', 0),
                species_scores.get('Agriculture_Specific', 0),
                species_scores.get('Rescue', 0),
                species_scores.get('Entertainment_Performance', 0),
                species_scores.get('Mining', 0),
                species_scores.get('Construction', 0),
                species_scores.get('Environmental_Monitoring', 0)
            ])
            
            features.append(feature_vector)
        
        # Create 2D visualization using PCA or t-SNE
        # For simplicity, we'll use the first two features
        x_coords = [f[0] for f in features]  # Physical score
        y_coords = [f[1] for f in features]  # Virtual score
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=robot_names,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Physical: %{x}<br>Virtual: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Robot Clusters by Classification",
            xaxis_title="Physical Score",
            yaxis_title="Virtual Score",
            plot_bgcolor='white',
            width=1200,
            height=800
        )
        
        return fig
    
    def save_cluster_as_png(self, filename: str = 'robot_clusters.png'):
        """
        Save the cluster visualization as a PNG file
        """
        fig = self.create_cluster_visualization()
        fig.write_image(f"./data/{filename}", width=1200, height=800)
        print(f"✅ Cluster visualization saved as PNG: data/{filename}")
    
    def save_taxonomy_bar_charts_as_png(self, filename: str = 'robot_taxonomy_bars.png'):
        """
        Save the taxonomy bar charts as a PNG file
        """
        fig = self.create_taxonomy_bar_charts()
        fig.write_image(f"./data/{filename}", width=1400, height=1200)
        print(f"✅ Taxonomy bar charts saved as PNG: data/{filename}")
    
    def save_simplified_tree_as_png(self, filename: str = 'robot_simplified_tree.png'):
        """
        Save the simplified tree visualization as a PNG file
        """
        fig = self.create_simplified_tree()
        fig.write_image(f"./data/{filename}", width=1400, height=1000)
        print(f"✅ Simplified tree visualization saved as PNG: data/{filename}")
    
    def save_taxonomy_summary_as_png(self, filename: str = 'robot_taxonomy_summary.png'):
        """
        Save the taxonomy summary chart as a PNG file
        """
        fig = self.create_taxonomy_summary_chart()
        fig.write_image(f"./data/{filename}", width=1200, height=800)
        print(f"✅ Taxonomy summary chart saved as PNG: data/{filename}")

    def create_dashboard(self) -> dash.Dash:
        """
        Create an interactive dashboard for exploring the robot taxonomy
        """
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Robot Taxonomy Tree of Life", className="text-center mb-4"),
                    html.P("Explore the evolutionary relationships between different types of robots", 
                           className="text-center text-muted")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Taxonomy Tree"),
                        dbc.CardBody([
                            dcc.Graph(id='tree-graph', style={'height': '600px'})
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Statistics"),
                        dbc.CardBody(id='stats-content')
                    ]),
                    
                    dbc.Card([
                        dbc.CardHeader("Filters"),
                        dbc.CardBody([
                            html.Label("Domain:"),
                            dcc.Dropdown(
                                id='domain-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Physical', 'value': 'Physical'},
                                    {'label': 'Virtual', 'value': 'Virtual'},
                                    {'label': 'Hybrid', 'value': 'Hybrid'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Kingdom:"),
                            dcc.Dropdown(
                                id='kingdom-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Industrial', 'value': 'Industrial'},
                                    {'label': 'Service', 'value': 'Service'},
                                    {'label': 'Medical', 'value': 'Medical'},
                                    {'label': 'Military', 'value': 'Military'},
                                    {'label': 'Research', 'value': 'Research'},
                                    {'label': 'Entertainment', 'value': 'Entertainment'},
                                    {'label': 'Agriculture', 'value': 'Agriculture'},
                                    {'label': 'Space', 'value': 'Space'},
                                    {'label': 'Marine', 'value': 'Marine'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Phylum:"),
                            dcc.Dropdown(
                                id='phylum-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Manipulator', 'value': 'Manipulator'},
                                    {'label': 'Mobile', 'value': 'Mobile'},
                                    {'label': 'Humanoid', 'value': 'Humanoid'},
                                    {'label': 'Modular', 'value': 'Modular'},
                                    {'label': 'Swarm', 'value': 'Swarm'},
                                    {'label': 'Soft', 'value': 'Soft'},
                                    {'label': 'Hybrid_Morphology', 'value': 'Hybrid_Morphology'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Class:"),
                            dcc.Dropdown(
                                id='class-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Static', 'value': 'Static'},
                                    {'label': 'Wheeled', 'value': 'Wheeled'},
                                    {'label': 'Legged', 'value': 'Legged'},
                                    {'label': 'Flying', 'value': 'Flying'},
                                    {'label': 'Swimming', 'value': 'Swimming'},
                                    {'label': 'Morphing', 'value': 'Morphing'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Order:"),
                            dcc.Dropdown(
                                id='order-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Teleoperated', 'value': 'Teleoperated'},
                                    {'label': 'Semi_Autonomous', 'value': 'Semi_Autonomous'},
                                    {'label': 'Autonomous', 'value': 'Autonomous'},
                                    {'label': 'Collaborative', 'value': 'Collaborative'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Family:"),
                            dcc.Dropdown(
                                id='family-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Vision_Based', 'value': 'Vision_Based'},
                                    {'label': 'LiDAR_Based', 'value': 'LiDAR_Based'},
                                    {'label': 'Tactile_Based', 'value': 'Tactile_Based'},
                                    {'label': 'Multimodal', 'value': 'Multimodal'},
                                    {'label': 'Minimal_Sensing', 'value': 'Minimal_Sensing'},
                                    {'label': 'GPS_Navigation', 'value': 'GPS_Navigation'},
                                    {'label': 'Acoustic_Based', 'value': 'Acoustic_Based'},
                                    {'label': 'Chemical_Sensing', 'value': 'Chemical_Sensing'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Genus:"),
                            dcc.Dropdown(
                                id='genus-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Electric', 'value': 'Electric'},
                                    {'label': 'Hydraulic', 'value': 'Hydraulic'},
                                    {'label': 'Pneumatic', 'value': 'Pneumatic'},
                                    {'label': 'Hybrid_Actuation', 'value': 'Hybrid_Actuation'},
                                    {'label': 'Smart_Materials', 'value': 'Smart_Materials'},
                                    {'label': 'Bio_Hybrid', 'value': 'Bio_Hybrid'},
                                    {'label': 'Passive', 'value': 'Passive'},
                                    {'label': 'Magnetic', 'value': 'Magnetic'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Species:"),
                            dcc.Dropdown(
                                id='species-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Assembly', 'value': 'Assembly'},
                                    {'label': 'Inspection', 'value': 'Inspection'},
                                    {'label': 'Transport', 'value': 'Transport'},
                                    {'label': 'Surgery', 'value': 'Surgery'},
                                    {'label': 'Exploration', 'value': 'Exploration'},
                                    {'label': 'Maintenance', 'value': 'Maintenance'},
                                    {'label': 'Security', 'value': 'Security'},
                                    {'label': 'Education', 'value': 'Education'},
                                    {'label': 'Companionship', 'value': 'Companionship'},
                                    {'label': 'Agriculture_Specific', 'value': 'Agriculture_Specific'},
                                    {'label': 'Rescue', 'value': 'Rescue'},
                                    {'label': 'Entertainment_Performance', 'value': 'Entertainment_Performance'},
                                    {'label': 'Mining', 'value': 'Mining'},
                                    {'label': 'Construction', 'value': 'Construction'},
                                    {'label': 'Environmental_Monitoring', 'value': 'Environmental_Monitoring'}
                                ],
                                value='all'
                            )
                        ])
                    ])
                ], width=4)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Robot Details"),
                        dbc.CardBody(id='robot-details')
                    ])
                ])
            ])
        ], fluid=True)
        
        @app.callback(
            Output('tree-graph', 'figure'),
            [Input('domain-filter', 'value'),
             Input('kingdom-filter', 'value'),
             Input('phylum-filter', 'value'),
             Input('class-filter', 'value'),
             Input('order-filter', 'value'),
             Input('family-filter', 'value'),
             Input('genus-filter', 'value'),
             Input('species-filter', 'value')]
        )
        def update_tree(domain_filter, kingdom_filter, phylum_filter, class_filter, order_filter, family_filter, genus_filter, species_filter):
            # Rebuild graph with filters
            self.build_taxonomy_tree()
            
            # Apply filters (simplified - in a real implementation, you'd filter the graph)
            fig = self.create_radial_tree_of_life()
            return fig
        
        @app.callback(
            Output('stats-content', 'children'),
            [Input('tree-graph', 'figure')]
        )
        def update_stats(figure):
            stats = [
                html.P(f"Total Robots: {len(self.robots_data)}"),
                html.P(f"Total Taxonomy Nodes: {len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'taxonomy'])}"),
                html.P(f"Total Connections: {len(self.graph.edges())}")
            ]
            return stats
        
        @app.callback(
            Output('robot-details', 'children'),
            [Input('tree-graph', 'clickData')]
        )
        def display_robot_details(click_data):
            if not click_data:
                return "Click on a robot node to see details"
            
            point = click_data['points'][0]
            node_name = point['text']
            
            # Find robot data
            for robot in self.robots_data:
                if robot['name'] == node_name:
                    return [
                        html.H5(robot['name']),
                        html.P(f"Manufacturer: {robot.get('manufacturer', 'Unknown')}"),
                        html.P(f"Description: {robot.get('description', 'No description available')}"),
                        html.H6("Classification:"),
                        html.Ul([
                            html.Li(f"Domain: {', '.join(robot.get('classification', {}).get('Domain', {}).keys())}"),
                            html.Li(f"Kingdom: {', '.join(robot.get('classification', {}).get('Kingdom', {}).keys())}"),
                            html.Li(f"Phylum: {', '.join(robot.get('classification', {}).get('Phylum', {}).keys())}"),
                            html.Li(f"Class: {', '.join(robot.get('classification', {}).get('Class', {}).keys())}"),
                            html.Li(f"Order: {', '.join(robot.get('classification', {}).get('Order', {}).keys())}"),
                            html.Li(f"Family: {', '.join(robot.get('classification', {}).get('Family', {}).keys())}"),
                            html.Li(f"Genus: {', '.join(robot.get('classification', {}).get('Genus', {}).keys())}"),
                            html.Li(f"Species: {', '.join(robot.get('classification', {}).get('Species', {}).keys())}")
                        ])
                    ]
            
            return f"Details for {node_name}"
        
        return app

if __name__ == "__main__":
    visualizer = RobotTreeVisualizer()
    visualizer.load_data()
    
    if visualizer.robots_data:
        # Create PNG visualizations
        visualizer.save_radial_tree_as_png()
        visualizer.save_phylogenetic_tree_as_png()
        visualizer.save_dendrogram_as_png()
        visualizer.save_cluster_as_png()
        visualizer.save_taxonomy_bar_charts_as_png()
        visualizer.save_simplified_tree_as_png()
        visualizer.save_taxonomy_summary_as_png()
        print("✅ PNG visualizations created successfully!")
    else:
        print("No robot data found. Please run the scraper and classifier first.") 