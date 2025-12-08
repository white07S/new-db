WE WILL BE WORKING INSIDE project folder. We are working on Conversational AI Agent System.

This is uv based python project. 

so use uv add lib name and uv run to run the code.

always use absolute python package import style. no relative path import. 

Anytime you need to know code syntax of a lib always just refer to installed path in venv and look into the function signature.

When you have issue with retreval code around doc_rag, sql_rag refer here for idea, source info: https://github.com/eosphoros-ai/DB-GPT/tree/main /  http://docs.dbgpt.cn/docs/overview/ 

When ever possible write integration test (with real providers), rather than mock fixture tests.  

Refer agents/modules folder for markdown files for whole system truth, implemetation guide and common_module_details.
common_module_details is important piece of info where you have important links of system which we want to reference in this project.

Our by default motivation is from here: https://github.com/openai/openai-agents-python/tree/main/docs and [!IMPORTANT FOR SYSTEM DESIGN IDEAS]https://github.com/openai/codex and 
https://github.com/Canner/WrenAI/tree/main/wren-ai-service/src/pipelines