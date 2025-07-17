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
        Build a hierarchical tree structure based on the taxonomy
        """
        # Define the taxonomy hierarchy
        taxonomy = {
            "Robots": {
                "Domain": {
                    "Industrial": {
                        "Manufacturing": ["Assembly", "Welding", "Painting", "Material Handling"],
                        "Logistics": ["Warehouse", "Sorting", "Packaging"],
                        "Construction": ["Demolition", "Building", "Inspection"]
                    },
                    "Service": {
                        "Healthcare": ["Surgical", "Rehabilitation", "Diagnostic"],
                        "Domestic": ["Cleaning", "Cooking", "Security"],
                        "Commercial": ["Retail", "Hospitality", "Education"]
                    },
                    "Research": {
                        "Exploration": ["Space", "Underwater", "Aerial"],
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
                        "Legged": ["Bipedal", "Quadrupedal", "Hexapod"],
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
            elif isinstance(value, list):
                for item in value:
                    item_id = f"{node_id}.{item}"
                    self.graph.add_node(item_id, type="category", level=len(item_id.split('.'))-1)
                    self.graph.add_edge(node_id, item_id)
    
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
            
            # Connect to domain
            for domain, score in classification.get('Domain', {}).items():
                if score > 0.5:
                    domain_node = f"Robots.Domain.{domain}"
                    if domain_node in self.graph:
                        self.graph.add_edge(domain_node, robot_id, weight=score)
            
            # Connect to mobility
            for mobility, score in classification.get('Mobility', {}).items():
                if score > 0.5:
                    mobility_node = f"Robots.Mobility.{mobility}"
                    if mobility_node in self.graph:
                        self.graph.add_edge(mobility_node, robot_id, weight=score)
            
            # Connect to autonomy
            for autonomy, score in classification.get('Autonomy', {}).items():
                if score > 0.5:
                    autonomy_node = f"Robots.Autonomy.{autonomy}"
                    if autonomy_node in self.graph:
                        self.graph.add_edge(autonomy_node, robot_id, weight=score)
            
            # Connect to size
            for size, score in classification.get('Size', {}).items():
                if score > 0.5:
                    size_node = f"Robots.Size.{size}"
                    if size_node in self.graph:
                        self.graph.add_edge(size_node, robot_id, weight=score)
    
    def create_tree_visualization(self) -> go.Figure:
        """
        Create a tree-like visualization of the robot taxonomy
        """
        if not self.graph.nodes():
            self.build_taxonomy_tree()
        
        # Calculate positions using hierarchical layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Separate nodes by type
        taxonomy_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'taxonomy']
        category_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'category']
        robot_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'robot']
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))
        
        # Add taxonomy nodes
        if taxonomy_nodes:
            tax_x = [pos[node][0] for node in taxonomy_nodes]
            tax_y = [pos[node][1] for node in taxonomy_nodes]
            fig.add_trace(go.Scatter(
                x=tax_x, y=tax_y,
                mode='markers+text',
                marker=dict(size=20, color='blue', symbol='diamond'),
                text=[node.split('.')[-1] for node in taxonomy_nodes],
                textposition="middle center",
                name='Taxonomy',
                hovertemplate='<b>%{text}</b><br>Type: Taxonomy<extra></extra>'
            ))
        
        # Add category nodes
        if category_nodes:
            cat_x = [pos[node][0] for node in category_nodes]
            cat_y = [pos[node][1] for node in category_nodes]
            fig.add_trace(go.Scatter(
                x=cat_x, y=cat_y,
                mode='markers+text',
                marker=dict(size=15, color='green', symbol='circle'),
                text=[node.split('.')[-1] for node in category_nodes],
                textposition="middle center",
                name='Categories',
                hovertemplate='<b>%{text}</b><br>Type: Category<extra></extra>'
            ))
        
        # Add robot nodes
        if robot_nodes:
            robot_x = [pos[node][0] for node in robot_nodes]
            robot_y = [pos[node][1] for node in robot_nodes]
            robot_names = [self.graph.nodes[node]['name'] for node in robot_nodes]
            robot_descriptions = [self.graph.nodes[node].get('description', '')[:100] + '...' 
                               if len(self.graph.nodes[node].get('description', '')) > 100 
                               else self.graph.nodes[node].get('description', '') 
                               for node in robot_nodes]
            
            fig.add_trace(go.Scatter(
                x=robot_x, y=robot_y,
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='circle'),
                text=robot_names,
                textposition="middle center",
                name='Robots',
                hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
                customdata=robot_descriptions
            ))
        
        fig.update_layout(
            title="Robot Taxonomy Tree of Life",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
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
                                    {'label': 'Industrial', 'value': 'Industrial'},
                                    {'label': 'Service', 'value': 'Service'},
                                    {'label': 'Research', 'value': 'Research'},
                                    {'label': 'Military', 'value': 'Military'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Mobility:"),
                            dcc.Dropdown(
                                id='mobility-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Stationary', 'value': 'Stationary'},
                                    {'label': 'Mobile', 'value': 'Mobile'}
                                ],
                                value='all'
                            ),
                            
                            html.Br(),
                            
                            html.Label("Autonomy:"),
                            dcc.Dropdown(
                                id='autonomy-filter',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Teleoperated', 'value': 'Teleoperated'},
                                    {'label': 'Semi-autonomous', 'value': 'Semi-autonomous'},
                                    {'label': 'Autonomous', 'value': 'Autonomous'}
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
             Input('mobility-filter', 'value'),
             Input('autonomy-filter', 'value')]
        )
        def update_tree(domain_filter, mobility_filter, autonomy_filter):
            # Rebuild graph with filters
            self.build_taxonomy_tree()
            
            # Apply filters (simplified - in a real implementation, you'd filter the graph)
            fig = self.create_tree_visualization()
            return fig
        
        @app.callback(
            Output('stats-content', 'children'),
            [Input('tree-graph', 'figure')]
        )
        def update_stats(figure):
            stats = [
                html.P(f"Total Robots: {len(self.robots_data)}"),
                html.P(f"Total Taxonomy Nodes: {len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'taxonomy'])}"),
                html.P(f"Total Categories: {len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'category'])}"),
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
                            html.Li(f"Mobility: {', '.join(robot.get('classification', {}).get('Mobility', {}).keys())}"),
                            html.Li(f"Autonomy: {', '.join(robot.get('classification', {}).get('Autonomy', {}).keys())}"),
                            html.Li(f"Size: {', '.join(robot.get('classification', {}).get('Size', {}).keys())}")
                        ])
                    ]
            
            return f"Details for {node_name}"
        
        return app
    
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
                domain_scores.get('Industrial', 0),
                domain_scores.get('Service', 0),
                domain_scores.get('Research', 0),
                domain_scores.get('Military', 0)
            ])
            
            # Mobility features
            mobility_scores = robot.get('classification', {}).get('Mobility', {})
            feature_vector.extend([
                mobility_scores.get('Stationary', 0),
                mobility_scores.get('Mobile', 0)
            ])
            
            # Autonomy features
            autonomy_scores = robot.get('classification', {}).get('Autonomy', {})
            feature_vector.extend([
                autonomy_scores.get('Teleoperated', 0),
                autonomy_scores.get('Semi-autonomous', 0),
                autonomy_scores.get('Autonomous', 0)
            ])
            
            features.append(feature_vector)
        
        # Create 2D visualization using PCA or t-SNE
        # For simplicity, we'll use the first two features
        x_coords = [f[0] for f in features]  # Industrial score
        y_coords = [f[1] for f in features]  # Service score
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=robot_names,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Industrial: %{x}<br>Service: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Robot Clusters by Classification",
            xaxis_title="Industrial Score",
            yaxis_title="Service Score",
            plot_bgcolor='white'
        )
        
        return fig

if __name__ == "__main__":
    visualizer = RobotTreeVisualizer()
    visualizer.load_data()
    
    if visualizer.robots_data:
        # Create tree visualization
        fig = visualizer.create_tree_visualization()
        fig.show()
        
        # Create cluster visualization
        cluster_fig = visualizer.create_cluster_visualization()
        cluster_fig.show()
        
        # Create dashboard
        app = visualizer.create_dashboard()
        app.run_server(debug=True, port=8050)
    else:
        print("No robot data found. Please run the scraper and classifier first.") 