from datetime import datetime
from pathlib import Path
import re

from ddgs import DDGS
import wikipedia
from pypdf import PdfReader
from langchain_core.tools import tool


DOCUMENTS_DIR = Path("documents")


def clean_text(text: str) -> str:
    """Clean messy PDF text."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Split long text into smaller chunks.

    Why?
    LLMs and search tools work better with smaller pieces of text.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def keyword_score(query: str, text: str) -> int:
    """
    Very simple local retrieval scoring.

    It counts how many query words appear in the PDF chunk.
    This is not advanced vector search yet, but it is simple and works.
    """
    query_words = set(query.lower().split())
    text_lower = text.lower()

    score = 0
    for word in query_words:
        if len(word) > 2 and word in text_lower:
            score += 1

    return score


@tool
def search_tool(query: str) -> str:
    """Search the web for current or recent information and return results with URLs."""
    results = []

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=5):
            title = result.get("title", "")
            href = result.get("href", "")
            body = result.get("body", "")

            results.append(
                f"Title: {title}\nURL: {href}\nSnippet: {body}"
            )

    if not results:
        return "No search results found."

    return "\n\n".join(results)

@tool
def image_search_tool(query: str) -> str:
    """Search for image URLs related to a query. Return up to 3 image URLs."""
    image_results = []

    with DDGS() as ddgs:
        for result in ddgs.images(query, max_results=3):
            title = result.get("title", "")
            image_url = result.get("image", "")
            source_url = result.get("url", "")

            if image_url:
                image_results.append(
                    f"Title: {title}\nImage URL: {image_url}\nSource URL: {source_url}"
                )

    if not image_results:
        return "No image results found."

    return "\n\n".join(image_results)


@tool
def wiki_tool(query: str) -> str:
    """Search Wikipedia for background information about a topic."""
    try:
        page_titles = wikipedia.search(query, results=3)

        if not page_titles:
            return "No Wikipedia results found."

        page = wikipedia.page(page_titles[0], auto_suggest=False)
        summary = wikipedia.summary(page.title, sentences=5, auto_suggest=False)

        return f"Title: {page.title}\nURL: {page.url}\nSummary: {summary}"

    except Exception as e:
        return f"Wikipedia search failed: {e}"


@tool
def pdf_research_tool(query: str) -> str:
    """
    Search local PDF documents inside the documents folder.

    Use this when the user asks about uploaded/local documents, PDFs, reports,
    lecture notes, CVs, papers, contracts, or files.
    """
    if not DOCUMENTS_DIR.exists():
        return "No documents folder found. Create a folder named 'documents' and put PDFs inside it."

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        return "No PDF files found in the documents folder."

    all_matches = []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))

            full_text = ""
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = clean_text(page_text)

                if page_text:
                    chunks = split_text_into_chunks(page_text)

                    for chunk in chunks:
                        score = keyword_score(query, chunk)

                        if score > 0:
                            all_matches.append(
                                {
                                    "score": score,
                                    "file": pdf_path.name,
                                    "page": page_number,
                                    "text": chunk
                                }
                            )

        except Exception as e:
            all_matches.append(
                {
                    "score": 0,
                    "file": pdf_path.name,
                    "page": "unknown",
                    "text": f"Could not read this PDF: {e}"
                }
            )

    if not all_matches:
        return "No relevant PDF content found for this query."

    all_matches.sort(key=lambda item: item["score"], reverse=True)
    top_matches = all_matches[:5]

    response_parts = []

    for match in top_matches:
        response_parts.append(
            f"File: {match['file']}\n"
            f"Page: {match['page']}\n"
            f"Relevant text: {match['text'][:1000]}"
        )

    return "\n\n---\n\n".join(response_parts)


@tool
def save_tool(data: str) -> str:
    """Save text data to a timestamped Markdown report file."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = reports_dir / f"research_report_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)

    return f"Research report saved to {filename}"