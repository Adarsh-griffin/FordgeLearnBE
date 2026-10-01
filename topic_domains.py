"""
Shared "which trusted sites should we prefer for this topic" logic, used by
both reference_links.py (web search) and topic_images.py (image search) so
the two stay in sync instead of drifting with their own copies.

Plain Serper results skew toward thin SEO-farm pages/hotlink-broken images;
these are sites students/teachers already recognize as reliable, split by
whether the topic is technical/programming (GeeksforGeeks, W3Schools) or
not (BYJU'S, Vedantu, Toppr, Aakash - popular Indian ed-tech platforms with
strong diagram/explainer content for school-level science, math, etc.).
"""

TECH_DOMAINS = ["geeksforgeeks.org", "w3schools.com"]
NON_TECH_DOMAINS = ["byjus.com", "vedantu.com", "toppr.com", "aakash.ac.in"]

TECH_KEYWORDS = (
    "programming", "code", "coding", "algorithm", "data structure", "python",
    "java", "javascript", "typescript", "html", "css", "sql", "database",
    "api", "software", "web development", "computer science", "dsa",
    "react", "node", "backend", "frontend", "machine learning", "oop",
    "operating system", "dbms", "networking", "git", "c++", "c program",
)


def is_tech_topic(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in TECH_KEYWORDS)


def preferred_domains(query: str) -> list:
    return TECH_DOMAINS if is_tech_topic(query) else NON_TECH_DOMAINS


def site_restricted_query(query: str, domains: list) -> str:
    site_filter = " OR ".join(f"site:{d}" for d in domains)
    return f"{query} ({site_filter})"
