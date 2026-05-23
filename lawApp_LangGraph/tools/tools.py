"""
Agent 工具集 — 网络搜索与 PDF 生成

工具列表:
    get_google_search   — SerpAPI 谷歌搜索,返回结构化结果(含 URL / 标题 / 摘要)
    fetch_webpage_text  — 抓取指定 URL 的网页正文文本
    markdown_to_pdf     — Markdown 转 PDF 文件
"""

import os
import time
from datetime import datetime
import markdown
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool

from lawApp_LangGraph.FastAPI.logging import tool as tool_log
from langsmith import traceable


# Tool 1: 谷歌搜索


@tool
@traceable(run_type="tool", name="tool_Google搜索")
def get_google_search(query: str) -> dict:
    """使用谷歌搜索API在线搜索法律相关信息.返回结构化结果,每项包含标题、链接、摘要.

    适用场景:
    - 法律案例库检索不足时,联网补充最新法规、司法解释
    - 查找特定法律条文的官方解释
    - 获取实时法律新闻和政策变动

    参数:
    query: 搜索关键词,中文或英文

    返回:
    dict,含 results 列表和格式化的 web_search_results:
    [{"title": "...", "link": "...", "snippet": "..."}, ...]
    """
    t0 = time.time()
    tool_log.info(
        "→ 调用工具: get_google_search",
        detail=f"query={query[:80]}",
    )

    search = SerpAPIWrapper()
    raw = search.results(query)

    structured = []
    for res in raw.get("organic_results", [])[:8]:
        structured.append(
            {
                "title": res.get("title", ""),
                "link": res.get("link", ""),
                "snippet": res.get("snippet", "")
            }
        )
    if not structured:
        tool_log.info(
            "← 工具返回: get_google_search",
            detail="未找到搜索结果",
            result=f"elapsed={time.time() - t0:.2f}s",
        )
        return {
            "status": "empty",
            "message": "未找到相关搜索结果",
            "web_search_results": [],
        }

    tool_log.info(
        "← 工具返回: get_google_search",
        detail=f"搜索结果{len(structured)}条",
        result=f"elapsed={time.time() - t0:.2f}s",
    )
    return {
        "status": "success",
        "count": len(structured),
        "web_search_results": structured
    }


# Tool 2: Markdown → PDF


def markdown_to_html(markdown_text: str) -> str:
    """将Markdown文本转换为HTML字符串,并启用表格等扩展功能"""
    return markdown.markdown(markdown_text, extensions=["extra", "codehilite"])


@tool
@traceable(run_type="tool", name="tool_Markdown转PDF")
def markdown_to_pdf(markdown_text: str, filename: str = None) -> dict:
    """MarkDown文件转为pdf,当用户指定pdf文件输出时使用.

    参数:
    markdown_text: markdown文本内容
    filename: 输出的pdf文件名(不含路径),默认为 report_{时间戳}.pdf

    返回:
    dict,含 pdf_path 和 is_pdf_output
    """
    import pdfkit

    t0 = time.time()
    tool_log.info(
        "→ 调用工具: markdown_to_pdf",
        detail=f"filename={filename or 'auto'} | content_len={len(markdown_text)}",
    )

    if not filename:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    html_content = markdown_to_html(markdown_text)

    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'SimHei', 'Microsoft YaHei', sans-serif; margin: 1cm; }}
            h1 {{ color: #333; }}
            code {{ font-family: monospace; background-color: #f4f4f4; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    options = {
        "page-size": "A4",
        "margin-top": "0.75in",
        "margin-right": "0.75in",
        "margin-bottom": "0.75in",
        "margin-left": "0.75in",
        "encoding": "UTF-8",
        "no-outline": None,
    }

    output_dir = "./pdf_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    pdfkit.from_string(styled_html, file_path, options=options)

    tool_log.info(
        "← 工具返回: markdown_to_pdf",
        detail=f"file={filename}",
        result=f"elapsed={time.time() - t0:.2f}s",
    )
    return {"pdf_path": file_path, "is_pdf_output": True}
