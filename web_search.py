from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 3):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    out = []
    for r in results:
        out.append({
            'title': r.get('title', ''),
            'snippet': r.get('body', ''),
            'url': r.get('href', '')
        })
    return out

def format_search_snippets(items):
    if not items:
        return ""
    lines = [f"- {x['title']} — {x['snippet']} (source: {x['url']})" for x in items]
    return "\n" + "\n".join(lines)
