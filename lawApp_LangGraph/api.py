from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import os
from typing import Optional

# 导入现有的图和工具
from lawApp_LangGraph.LangGraph_lawApp import graph
from lawApp_LangGraph.tools.tools import markdown_to_pdf

app = FastAPI(title="Legal Consultation API", description="基于LangGraph的法律咨询后端接口", version="1.0.0")

@app.post("/ask")
async def ask_question(query: str, output_pdf: Optional[bool] = False):
    """
    处理用户问答请求
    
    - **query**: 用户的问题
    - **output_pdf**: 是否输出为PDF格式 (默认False)
    
    返回:
    - 如果output_pdf=False: JSON格式的答案
    - 如果output_pdf=True: PDF文件下载
    """
    try:
        # 初始化最终状态
        final_state = None
        
        # 流式执行图
        for chunk in graph.stream({"query": query, "messages": []}):
            final_state = chunk
        
        # 获取最终答案
        answer = final_state.get("final_answer", "") if final_state else ""
        
        if not answer:
            raise HTTPException(status_code=500, detail="未能生成答案")
        
        if output_pdf:
            # 生成PDF文件
            pdf_filename = "legal_answer.pdf"
            pdf_path = markdown_to_pdf(answer, pdf_filename)
            
            # 检查PDF是否生成成功
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=500, detail="PDF生成失败")
            
            # 返回PDF文件
            return FileResponse(
                path=pdf_path,
                filename=pdf_filename,
                media_type='application/pdf'
            )
        else:
            # 返回JSON格式答案
            return {
                "query": query,
                "answer": answer,
                "output_format": "text"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

@app.get("/")
async def root():
    """API根路径"""
    return {"message": "法律咨询API服务运行中", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)