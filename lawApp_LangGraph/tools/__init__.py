from lawApp_LangGraph.tools.tools import (
    fetch_webpage_text,
    get_google_search,
    markdown_to_pdf,
)
from lawApp_LangGraph.tools.rag_tools import (
    retrieve_legal_knowledge,
    evaluate_case_relevance,
    analyze_legal_issue,
)

# Agent 可用的全部工具列表
ALL_TOOLS = [
    get_google_search,
    markdown_to_pdf,
    retrieve_legal_knowledge,
    evaluate_case_relevance,
    analyze_legal_issue,
]
