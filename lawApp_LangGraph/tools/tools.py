"""
Agent 工具集 — 网络搜索与 PDF 生成

工具列表:
    get_google_search   — SerpAPI 谷歌搜索,返回结构化结果(含 URL / 标题 / 摘要)
    fetch_webpage_text  — 抓取指定 URL 的网页正文文本
    markdown_to_pdf     — Markdown 转 PDF 文件
"""
import json
import os
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Dict, List
from urllib.request import Request, urlopen

import markdown
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool


# ============================================================================
# Tool 1: 谷歌搜索
# ============================================================================


@tool
def get_google_search(query: str) -> str:
    """使用谷歌搜索API在线搜索法律相关信息.返回结构化结果,每项包含标题、链接、摘要.

    适用场景:
    - 法律案例库检索不足时,联网补充最新法规、司法解释
    - 查找特定法律条文的官方解释
    - 获取实时法律新闻和政策变动

    参数:
    query: 搜索关键词,中文或英文

    返回:
    JSON 字符串,含 results 列表:
    [{"title": "...", "link": "...", "snippet": "..."}, ...]
    """
    search = SerpAPIWrapper()
    raw = search.results(query)

    structured = []
    for res in raw.get("organic_results", [])[:8]:
        structured.append(
            {
                "title": res.get("title", ""),
                "link": res.get("link", ""),
                "snippet": res.get("snippet", ""),
                "source": res.get("source", ""),
            }
        )

    if not structured:
        return json.dumps(
            {"status": "empty", "message": "未找到相关搜索结果", "results": []},
            ensure_ascii=False,
        )

    return json.dumps(
        {"status": "success", "count": len(structured), "results": structured},
        ensure_ascii=False,
        indent=2,
    )




# class _TextExtractor(HTMLParser):
#     """从 HTML 中提取纯文本,跳过 script/style 标签"""

#     def __init__(self):
#         super().__init__()
#         self.text_parts: List[str] = []
#         self.skip_tags = {"script", "style", "noscript", "meta", "link", "head"}
#         self._skip_depth = 0

#     def handle_starttag(self, tag, attrs):
#         if tag.lower() in self.skip_tags:
#             self._skip_depth += 1

#     def handle_endtag(self, tag):
#         if tag.lower() in self.skip_tags and self._skip_depth > 0:
#             self._skip_depth -= 1

#     def handle_data(self, data):
#         if self._skip_depth == 0:
#             text = data.strip()
#             if text and len(text) > 1:
#                 self.text_parts.append(text)


# def _extract_text_from_html(html: str, max_chars: int = 3000) -> str:
#     parser = _TextExtractor()
#     parser.feed(html)
#     raw = " ".join(parser.text_parts)
#     # 合并多余空白
#     raw = re.sub(r"\s+", " ", raw)
#     return raw[:max_chars]


# @tool
# def fetch_webpage_text(url: str, max_chars: int = 3000) -> str:
#     """抓取指定网址的网页正文文本.用于在谷歌搜索返回摘要后,进一步获取网页的完整内容.

#     适用场景:
#     - 谷歌搜索返回了相关的链接,需要查看完整内容
#     - 需要核实搜索结果摘要中的具体细节
#     - 获取法律条文原文或官方公告全文

#     参数:
#     url: 要抓取的网页链接(必须是完整的 http/https URL)
#     max_chars: 最大返回字符数,默认 3000

#     返回:
#     网页正文文本(纯文本,已去除 HTML 标签和脚本)
#     """
#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/125.0.0.0 Safari/537.36"
#         ),
#         "Accept": "text/html,application/xhtml+xml",
#         "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
#     }

#     try:
#         req = Request(url, headers=headers)
#         with urlopen(req, timeout=15) as resp:
#             content_type = resp.headers.get("Content-Type", "")
#             charset = "utf-8"
#             if "charset=" in content_type:
#                 charset = content_type.split("charset=")[-1].split(";")[0].strip()

#             html = resp.read().decode(charset, errors="replace")
#     except Exception as e:
#         return json.dumps(
#             {
#                 "status": "error",
#                 "url": url,
#                 "error": f"网页抓取失败: {str(e)}",
#                 "text": "",
#             },
#             ensure_ascii=False,
#         )

#     text = _extract_text_from_html(html, max_chars)

#     if not text.strip():
#         return json.dumps(
#             {"status": "empty", "url": url, "error": "未能提取到有效文本内容", "text": ""},
#             ensure_ascii=False,
#         )

#     return json.dumps(
#         {"status": "success", "url": url, "text": text, "length": len(text)},
#         ensure_ascii=False,
#         indent=2,
#     )



# Tool 2: Markdown → PDF

def markdown_to_html(markdown_text: str) -> str:
    """将Markdown文本转换为HTML字符串,并启用表格等扩展功能"""
    return markdown.markdown(markdown_text, extensions=["extra", "codehilite"])


@tool
def markdown_to_pdf(markdown_text: str, filename: str = None) -> str:
    """MarkDown文件转为pdf,当用户指定pdf文件输出时使用.

    参数:
    markdown_text: markdown文本内容
    filename: 输出的pdf文件名(不含路径),默认为 report_{时间戳}.pdf

    返回:
    文件存放路径
    """
    import pdfkit

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

    return f"PDF 已成功生成,文件路径:{file_path}"
