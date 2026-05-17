from datetime import datetime
from pathlib import Path
import re
import shutil

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from tools import search_tool, image_search_tool, wiki_tool, pdf_research_tool, save_tool

load_dotenv()


DOCUMENTS_DIR = Path("documents")
REPORTS_DIR = Path("reports")


class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the research.")
    output_style: str = Field(description="The selected output style.")
    summary: str = Field(description="The final research summary.")
    key_points: list[str] = Field(description="Important bullet points from the research.")
    sources: list[str] = Field(description="Sources, URLs, PDF names, or references used.")
    source_quality: list[str] = Field(description="Quality/reliability notes about the sources.")
    image_urls: list[str] = Field(description="A list of image URLs related to the research topic.")
    tools_used: list[str] = Field(description="The tools used by the agent.")


def create_safe_filename(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")

    if not text:
        text = "research_report"

    return text[:60]


def create_markdown_report(response: ResearchResponse) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    key_points_text = "\n".join([f"- {point}" for point in response.key_points])

    if response.sources:
        sources_text = "\n".join([f"- {source}" for source in response.sources])
    else:
        sources_text = "- No external sources provided."

    if response.source_quality:
        quality_text = "\n".join([f"- {item}" for item in response.source_quality])
    else:
        quality_text = "- No source quality notes provided."

    if response.image_urls:
        images_text = "\n".join([f"![Image]({url})" for url in response.image_urls[:3]])
    else:
        images_text = "- No images provided."

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

## Source Quality Check

{quality_text}

---

## Related Images

{images_text}

---

## Tools Used

{tools_text}
"""

    return markdown


def save_uploaded_pdfs(uploaded_files):
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    saved_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            file_path = DOCUMENTS_DIR / uploaded_file.name

            with open(file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)

            saved_files.append(uploaded_file.name)

    return saved_files


def save_markdown_report(response: ResearchResponse, markdown_report: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_topic = create_safe_filename(response.topic)

    filename = REPORTS_DIR / f"{safe_topic}_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdown_report)

    return filename


def build_agent():
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0
    )

    tools = [
        search_tool,
        image_search_tool,
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
- image_search_tool: for finding up to 3 relevant image URLs
- wiki_tool: for general background information
- pdf_research_tool: for searching local PDF documents
- save_tool: for saving text when explicitly useful or requested

Your task:
1. Understand the user's research question.
2. Follow the selected output style.
3. Use tools when useful.
4. If the user uploads PDFs or asks about PDFs/documents/files, use pdf_research_tool.
5. If the question needs current or recent information, use search_tool.
6. If the question needs general background, use wiki_tool.
7. If the research topic has a clear visual subject, use image_search_tool and include up to 3 image URLs in image_urls.
8. Return a structured response with:
   - topic
   - output_style
   - summary
   - key_points
   - sources
   - source_quality
   - image_urls
   - tools_used

Visual subject examples:
- animals
- vehicles
- tanks
- historical objects
- places
- people
- products
- diagrams
- architecture

Source quality rules:
- Official documentation, government pages, university pages, research papers, and company documentation are high quality.
- Wikipedia is useful for background but should be marked as medium quality.
- Random blogs, forums, and unknown websites should be marked as lower quality unless clearly credible.
- PDF documents from the user should be described as local/user-provided document sources.
- Do not invent sources or URLs.
"""
    )

    return agent


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Research Assistant")
st.write(
    "Research topics using web search, Wikipedia, local PDF documents, structured output, "
    "source quality checking, image retrieval, and Markdown report generation."
)

with st.sidebar:
    st.header("Settings")

    output_style = st.selectbox(
        "Output style",
        [
            "Short summary",
            "Detailed report",
            "Bullet-point notes",
            "Academic style",
            "Interview preparation"
        ]
    )

    use_web = st.checkbox("Allow web search", value=True)
    use_wiki = st.checkbox("Allow Wikipedia", value=True)
    use_pdf = st.checkbox("Allow PDF research", value=True)
    use_images = st.checkbox("Show related images", value=True)

    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        saved_files = save_uploaded_pdfs(uploaded_files)
        st.success(f"Saved PDFs: {', '.join(saved_files)}")


query = st.text_area(
    "What do you want to research?",
    placeholder="Example: Research Tiger tank history and show related images."
)

run_button = st.button("Run Research")

if run_button:
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        tool_instructions = []

        if use_web:
            tool_instructions.append("- Web search is allowed.")
        else:
            tool_instructions.append("- Do not use web search.")

        if use_wiki:
            tool_instructions.append("- Wikipedia is allowed.")
        else:
            tool_instructions.append("- Do not use Wikipedia.")

        if use_pdf:
            tool_instructions.append("- PDF research is allowed. Use pdf_research_tool if useful.")
        else:
            tool_instructions.append("- Do not use PDF research.")

        if use_images:
            tool_instructions.append(
                "- Image search is allowed. Use image_search_tool for visual topics and return up to 3 image URLs."
            )
        else:
            tool_instructions.append(
                "- Do not use image_search_tool. Return an empty image_urls list."
            )

        user_message = f"""
Research question:
{query}

Output style:
{output_style}

Tool permissions:
{chr(10).join(tool_instructions)}

Output style instructions:
- Short summary: Give a concise answer in simple language.
- Detailed report: Give a more complete explanation with sections.
- Bullet-point notes: Focus on clear bullet points.
- Academic style: Use a formal tone and explain concepts carefully.
- Interview preparation: Explain the topic in a way that helps someone discuss it in an interview.

Also evaluate the quality of each source you used.
"""

        with st.spinner("Researching..."):
            agent = build_agent()

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

        st.success("Research complete!")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Summary", "Sources", "Markdown Report", "Raw Structured Output"]
        )

        with tab1:
            st.subheader(response.topic)
            st.write(response.summary)

            st.subheader("Key Points")
            if response.key_points:
                for point in response.key_points:
                    st.write(f"- {point}")
            else:
                st.write("No key points returned.")

            st.subheader("Related Images")

            if response.image_urls:
                cols = st.columns(3)

                for index, image_url in enumerate(response.image_urls[:3]):
                    with cols[index]:
                        st.image(
                            image_url,
                            caption=f"Image {index + 1}",
                            use_container_width=True
                        )
            else:
                st.write("No images returned.")

            st.subheader("Tools Used")
            if response.tools_used:
                for tool in response.tools_used:
                    st.write(f"- {tool}")
            else:
                st.write("No tools used.")

        with tab2:
            st.subheader("Sources / Citations")

            if response.sources:
                for source in response.sources:
                    st.write(f"- {source}")
            else:
                st.write("No sources returned.")

            st.subheader("Source Quality Check")

            if response.source_quality:
                for quality in response.source_quality:
                    st.write(f"- {quality}")
            else:
                st.write("No source quality notes returned.")

        with tab3:
            st.subheader("Markdown Report")
            st.code(markdown_report, language="markdown")

            st.download_button(
                label="Download Markdown Report",
                data=markdown_report,
                file_name=report_path.name,
                mime="text/markdown",
                key=f"download_report_{report_path.name}"
            )

            st.info(f"Report also saved locally to: {report_path}")

        with tab4:
            st.json(response.model_dump())