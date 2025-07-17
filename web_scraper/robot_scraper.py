import requests
from bs4 import BeautifulSoup
import time
import json
import re
from typing import List, Dict, Any
from fake_useragent import UserAgent
import logging

class RobotScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.robots_data = []
        
    def search_robots(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search for robots using multiple search terms and sources
        """
        all_robots = []
        
        for term in search_terms:
            print(f"Searching for: {term}")
            robots = self._search_wikipedia(term)
            all_robots.extend(robots)
            
            robots = self._search_robot_database(term)
            all_robots.extend(robots)
            
            time.sleep(1)  # Be respectful to servers
            
        return all_robots
    
    def _search_wikipedia(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for robot-related articles
        """
        robots = []
        
        # Search for robot categories and lists
        search_urls = [
            f"https://en.wikipedia.org/wiki/Category:Robots",
            f"https://en.wikipedia.org/wiki/List_of_robots",
            f"https://en.wikipedia.org/wiki/Category:Industrial_robots",
            f"https://en.wikipedia.org/wiki/Category:Service_robots"
        ]
        
        for url in search_urls:
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find robot links
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link.get('href')
                        if href and '/wiki/' in href and not href.startswith('/wiki/Category:'):
                            title = link.get_text().strip()
                            if self._is_robot_related(title, search_term):
                                robot_info = self._extract_robot_info(href, title)
                                if robot_info:
                                    robots.append(robot_info)
                                    
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                
        return robots
    
    def _search_robot_database(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search robot databases and manufacturer websites
        """
        robots = []
        
        # Common robot manufacturer websites
        manufacturer_sites = [
            "https://www.abb.com/robotics",
            "https://www.fanuc.com/robots",
            "https://www.kuka.com/en-us/products/robotics-systems",
            "https://www.yaskawa.com/products/robotics"
        ]
        
        for site in manufacturer_sites:
            try:
                response = self.session.get(site)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract robot information from manufacturer sites
                    robot_info = self._extract_from_manufacturer_site(soup, site)
                    robots.extend(robot_info)
                    
            except Exception as e:
                print(f"Error scraping {site}: {e}")
                
        return robots
    
    def _is_robot_related(self, title: str, search_term: str) -> bool:
        """
        Check if a title is robot-related
        """
        robot_keywords = [
            'robot', 'robotic', 'automation', 'automated', 'mechanical',
            'industrial', 'service', 'humanoid', 'android', 'cyborg'
        ]
        
        title_lower = title.lower()
        search_lower = search_term.lower()
        
        return any(keyword in title_lower for keyword in robot_keywords) or search_lower in title_lower
    
    def _extract_robot_info(self, wiki_url: str, title: str) -> Dict[str, Any]:
        """
        Extract detailed information about a robot from Wikipedia
        """
        try:
            # Fix URL construction - ensure proper formatting
            if wiki_url.startswith('http'):
                full_url = wiki_url
            else:
                full_url = f"https://en.wikipedia.org{wiki_url}"
            response = self.session.get(full_url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract basic information
                robot_info = {
                    'name': title,
                    'url': full_url,
                    'description': '',
                    'category': '',
                    'manufacturer': '',
                    'year': '',
                    'applications': [],
                    'specifications': {}
                }
                
                # Extract description from first paragraph
                content = soup.find('div', {'id': 'mw-content-text'})
                if content:
                    paragraphs = content.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if text and len(text) > 50:
                            robot_info['description'] = text[:500] + "..." if len(text) > 500 else text
                            break
                
                # Extract infobox information
                infobox = soup.find('table', {'class': 'infobox'})
                if infobox:
                    rows = infobox.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            key = cells[0].get_text().strip().lower()
                            value = cells[1].get_text().strip()
                            
                            if 'manufacturer' in key:
                                robot_info['manufacturer'] = value
                            elif 'year' in key or 'introduced' in key:
                                robot_info['year'] = value
                            elif 'application' in key or 'use' in key:
                                robot_info['applications'].append(value)
                
                return robot_info
                
        except Exception as e:
            print(f"Error extracting robot info from {wiki_url}: {e}")
            
        return None
    
    def _extract_from_manufacturer_site(self, soup: BeautifulSoup, site_url: str) -> List[Dict[str, Any]]:
        """
        Extract robot information from manufacturer websites
        """
        robots = []
        
        # Look for robot product links
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            title = link.get_text().strip()
            
            if self._is_robot_related(title, 'robot'):
                robot_info = {
                    'name': title,
                    'url': href if href.startswith('http') else f"{site_url.rstrip('/')}/{href.lstrip('/')}",
                    'manufacturer': self._extract_manufacturer_from_url(site_url),
                    'description': '',
                    'category': 'industrial',
                    'year': '',
                    'applications': [],
                    'specifications': {}
                }
                robots.append(robot_info)
        
        return robots
    
    def _extract_manufacturer_from_url(self, url: str) -> str:
        """
        Extract manufacturer name from URL
        """
        domain = url.split('//')[1].split('/')[0]
        return domain.replace('www.', '').split('.')[0].title()
    
    def save_data(self, filename: str = 'robots_data.json'):
        """
        Save scraped robot data to JSON file
        """
        with open(f'./data/{filename}', 'w') as f:
            json.dump(self.robots_data, f, indent=2)
        print(f"Saved {len(self.robots_data)} robots to {filename}")

if __name__ == "__main__":
    scraper = RobotScraper()
    
    # Search terms for different types of robots
    search_terms = [
        "industrial robots",
        "service robots", 
        "humanoid robots",
        "medical robots",
        "military robots",
        "domestic robots",
        "educational robots"
    ]
    
    robots = scraper.search_robots(search_terms)
    scraper.robots_data = robots
    scraper.save_data()
    print(f"Found {len(robots)} robots") 