from datetime import datetime
from pathlib import Path
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from tools import search_tool, wiki_tool, pdf_research_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the research.")
    output_style: str = Field(description="The selected output style.")
    summary: str = Field(description="The final research summary.")
    key_points: list[str] = Field(description="Important bullet points from the research.")
    sources: list[str] = Field(description="Sources, URLs, PDF names, or references used.")
    tools_used: list[str] = Field(description="The tools used by the agent.")


def create_safe_filename(text: str) -> str:
    """
    Convert a topic into a safe filename.

    Example:
    'AI Agents in Healthcare?' -> 'ai_agents_in_healthcare'
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")

    if not text:
        text = "research_report"

    return text[:60]


def create_markdown_report(response: ResearchResponse) -> str:
    """
    Convert the structured AI response into a professional Markdown report.
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    key_points_text = "\n".join([f"- {point}" for point in response.key_points])

    if response.sources:
        sources_text = "\n".join([f"- {source}" for source in response.sources])
    else:
        sources_text = "- No external sources provided."

    if response.tools_used:
        tools_text = "\n".join([f"- {tool}" for tool in response.tools_used])
    else:
        tools_text = "- No tools used."

    markdown = f"""# {response.topic}

**Generated at:** {created_at}  
**Output style:** {response.output_style}

---

## Summary

{response.summary}

---

## Key Points

{key_points_text}

---

## Sources / Citations

{sources_text}

---

## Tools Used

{tools_text}
"""

    return markdown


def save_markdown_report(response: ResearchResponse, markdown_report: str) -> Path:
    """
    Save the Markdown report locally inside the reports folder.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_topic = create_safe_filename(response.topic)

    filename = reports_dir / f"{safe_topic}_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdown_report)

    return filename


def build_agent():
    """
    Create the LangChain agent.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0
    )

    tools = [
        search_tool,
        wiki_tool,
        pdf_research_tool,
        save_tool
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=ResearchResponse,
        system_prompt="""
You are an AI research assistant.

You can use these tools:
- search_tool: for recent or current web information
- wiki_tool: for general background information
- pdf_research_tool: for searching local PDF documents
- save_tool: for saving text when the user explicitly asks you to save something

Your task:
1. Understand the user's research question.
2. Follow the selected output style.
3. Use tools when useful.
4. If the question mentions PDFs, documents, files, lecture notes, papers, or local material, use pdf_research_tool.
5. If the question needs current or recent information, use search_tool.
6. If the question needs general background, use wiki_tool.
7. Return a structured response with:
   - topic
   - output_style
   - summary
   - key_points
   - sources
   - tools_used

Rules:
- Do not invent sources.
- If you use web search, include URLs in sources.
- If you use Wikipedia, include the Wikipedia page URL in sources.
- If you use PDF research, include PDF file names and page numbers in sources.
- Mention every tool you used in tools_used.
"""
    )

    return agent


def main():
    print("\nAI Research Assistant")
    print("---------------------")

    query = input("\nWhat do you want to research? ")

    print("\nChoose output style:")
    print("1. Short summary")
    print("2. Detailed report")
    print("3. Bullet-point notes")
    print("4. Academic style")
    print("5. Interview preparation")

    style_choice = input("\nEnter number 1-5: ")

    output_styles = {
        "1": "Short summary",
        "2": "Detailed report",
        "3": "Bullet-point notes",
        "4": "Academic style",
        "5": "Interview preparation"
    }

    output_style = output_styles.get(style_choice, "Detailed report")

    user_message = f"""
Research question:
{query}

Output style:
{output_style}

Instructions for output style:
- Short summary: Give a concise answer in simple language.
- Detailed report: Give a more complete explanation with sections.
- Bullet-point notes: Focus on clear bullet points.
- Academic style: Use a formal tone and explain concepts carefully.
- Interview preparation: Explain the topic in a way that helps someone discuss it in an interview.

Use PDF research if the question asks about local documents, PDFs, papers, lecture notes, files, or uploaded documents.
"""

    agent = build_agent()

    print("\nResearching...\n")

    raw_response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
    )

    response = raw_response["structured_response"]

    markdown_report = create_markdown_report(response)
    report_path = save_markdown_report(response, markdown_report)

    print("\n--- Topic ---")
    print(response.topic)

    print("\n--- Summary ---")
    print(response.summary)

    print("\n--- Key Points ---")
    for point in response.key_points:
        print(f"- {point}")

    print("\n--- Sources / Citations ---")
    for source in response.sources:
        print(f"- {source}")

    print("\n--- Tools Used ---")
    for tool in response.tools_used:
        print(f"- {tool}")

    print("\n--- Markdown Report Saved ---")
    print(report_path)


if __name__ == "__main__":
    main()