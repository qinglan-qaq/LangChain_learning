from lawApp_LangGraph.tools.tools import (
    get_google_search,
    markdown_to_pdf,
)
from lawApp_LangGraph.tools.rag_tools import (
    retrieve_legal_knowledge,
    evaluate_case_relevance,
    analyze_legal_issue,
)
from lawApp_LangGraph.tools.db_tools import (
    search_memory,
    save_to_memory,
    fetch_laws,
)

# Agent 可用的全部工具列表
ALL_TOOLS = [
    search_memory,
    save_to_memory,
    fetch_laws,
    get_google_search,
    markdown_to_pdf,
    retrieve_legal_knowledge,
    evaluate_case_relevance,
    analyze_legal_issue,
]
