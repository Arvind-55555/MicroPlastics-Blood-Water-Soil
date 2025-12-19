"""Web scraping utilities for data sources without APIs."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
import time
from loguru import logger


class WebScraper:
    """Generic web scraper with robots.txt compliance."""
    
    def __init__(self, base_url: str, output_dir: Path, delay: float = 1.0):
        self.base_url = base_url
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; MicroplasticML/1.0; +https://example.com/bot)'
        })
        
        # Check robots.txt
        self.robots_parser = RobotFileParser()
        robots_url = urljoin(base_url, '/robots.txt')
        try:
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            logger.info(f"Loaded robots.txt from {robots_url}")
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {e}")
            self.robots_parser = None
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if self.robots_parser is None:
            return True
        return self.robots_parser.can_fetch(self.session.headers['User-Agent'], url)
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page."""
        if not self.can_fetch(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return None
        
        try:
            time.sleep(self.delay)  # Be respectful
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def find_download_links(self, url: str, patterns: List[str]) -> List[Dict[str, str]]:
        """Find download links matching patterns."""
        soup = self.fetch_page(url)
        if soup is None:
            return []
        
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            # Check if link matches any pattern
            for pattern in patterns:
                if pattern.lower() in href.lower() or pattern.lower() in full_url.lower():
                    links.append({
                        "url": full_url,
                        "text": link.get_text(strip=True),
                        "pattern": pattern
                    })
        
        return links
    
    def download_file(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download a file from URL."""
        if not self.can_fetch(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return None
        
        try:
            time.sleep(self.delay)
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            if filename is None:
                filename = url.split('/')[-1] or "download"
            
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {filename} from {url}")
            return filepath
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

