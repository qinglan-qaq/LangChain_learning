import os
from datetime import datetime
import markdown
import pdfkit
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool

@tool
def get_google_search(query: str):
    """使用谷歌搜索API在线搜索信息,适用于回答时事、确认事实或寻找特定网址"""

    # 环境变量中已设置 SERPAPI_API_KEY
    search = SerpAPIWrapper()

    # 获取结构化结果
    results = search.results(query)

    # 格式化输出，方便 LLM 阅读
    formatted_results = []
    for res in results.get("organic_results", [])[:5]:
        formatted_results.append(
            f"标题: {res.get('title')}\n摘要: {res.get('snippet')}\n链接: {res.get('link')}\n---"
        )
    if not formatted_results:
        return "未找到相关搜索结果。"

    return "\n".join(formatted_results)


def markdown_to_html(markdown_text: str) -> str:
    """将Markdown文本转换为HTML字符串，并启用表格等扩展功能"""
    # 'extra' 扩展支持了表格、围栏代码块等常用Markdown语法

    return markdown.markdown(markdown_text, extensions=['extra', 'codehilite'])


# MarkDown文件转为pdf
@tool
def markdown_to_pdf(markdown_text: str, filename: str = None) -> str:
    """
    MarkDown文件转为pdf,当用户指定pdf文件输出时使用

    参数:
    markdown文本 , 文件名

    返回:
    返回文件存放路径

    :param markdown_text:
    :param filename:
    :return:
    """
    # 生成文件名
    if not filename:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    # 将 Markdown 转换为 HTML
    html_content = markdown_to_html(markdown_text)

    # --- 添加 CSS 样式，解决中文乱码和排版问题 ---
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

    # 配置 PDF 选项
    options = {
        'page-size': 'A4',  # 页面大小
        'margin-top': '0.75in',  # 上边距
        'margin-right': '0.75in',  # 右边距
        'margin-bottom': '0.75in',  # 下边距
        'margin-left': '0.75in',  # 左边距
        'encoding': "UTF-8",  # 编码
        'no-outline': None  # 无轮廓
    }

    # 确保输出目录存在
    output_dir = "./pdf_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # 将HTML字符串转换为PDF
    pdfkit.from_string(styled_html, file_path, options=options)

    return f"PDF 已成功生成，文件路径：{file_path}"


@tool
def send_email(email_address: str = None) -> str:
    """当用户要求发送邮件时
    可以使用该工具发送消息

    :param email_address:str
    :return: 返回成功或者失败消息
    """



