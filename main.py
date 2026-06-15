import ast
import json
import operator
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Mini AI Backend with DeepSeek")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ClassifyRequest(BaseModel):
    message: str = Field(..., min_length=1)


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)


@app.get("/")
def home():
    return {
        "message": "Mini AI Backend with DeepSeek is running",
        "endpoints": [
            "GET /",
            "POST /analyze",
            "POST /chat",
            "POST /classify",
            "POST /agent"
        ]
    }


@app.post("/analyze")
def analyze_text(request: TextRequest):
    words = request.text.split()

    return {
        "original_text": request.text,
        "word_count": len(words),
        "char_count": len(request.text),
        "summary": f"This text has {len(words)} words and {len(request.text)} characters."
    }


@app.post("/chat")
def chat(request: ChatRequest):
    check_api_key()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Explain things clearly and simply."
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            stream=False,
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek chat request failed: {str(e)}"
        )


@app.post("/classify")
def classify_message(request: ClassifyRequest):
    check_api_key()

    system_prompt = """
You are an intent classification engine.

Classify the user's message into exactly one intent.

Available intents:
- qa: user asks a question
- summarization: user wants to summarize content
- translation: user wants translation
- coding: user asks for code or debugging
- writing: user wants writing or rewriting
- calculation: user asks for math calculation
- text_analysis: user wants text statistics or analysis
- unknown: unclear intent

You must output valid json only.

Example JSON output:
{
  "intent": "summarization",
  "confidence": 0.92,
  "reason": "The user asks to summarize a document."
}
"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            stream=False,
        )

        raw_content = response.choices[0].message.content

        if not raw_content:
            raise HTTPException(
                status_code=500,
                detail="Model returned empty content."
            )

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Model did not return valid JSON.",
                    "raw_content": raw_content
                }
            )

        return parsed

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek structured output failed: {str(e)}"
        )


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a math expression. Use this when the user asks for arithmetic calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, for example: 23 * 19 + 7"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_counter",
            "description": "Count words and characters in a given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze."
                    }
                },
                "required": ["text"]
            }
        }
    }
]


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


@app.post("/agent")
def run_agent(request: AgentRequest):
    check_api_key()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-using AI agent. "
                "When the user asks for arithmetic calculation, use calculator. "
                "When the user asks for word or character counting, use word_counter. "
                "If no tool is needed, answer directly. "
                "After receiving tool results, answer clearly and briefly."
            )
        },
        {
            "role": "user",
            "content": request.message
        }
    ]

    try:
        first_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            stream=False,
        )

        assistant_message = first_response.choices[0].message

        if not assistant_message.tool_calls:
            return {
                "mode": "direct_answer",
                "reply": assistant_message.content,
                "tool_results": []
            }

        assistant_tool_calls = []

        for tool_call in assistant_message.tool_calls:
            assistant_tool_calls.append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })

        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": assistant_tool_calls
        })

        tool_results = []

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            result = execute_tool(tool_name, tool_args)
            tool_results.append(result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

        final_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            stream=False,
        )

        return {
            "mode": "tool_used",
            "reply": final_response.choices[0].message.content,
            "tool_results": tool_results
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )


def check_api_key():
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is missing. Please set it in your .env file."
        )


def execute_tool(tool_name: str, arguments: dict[str, Any]):
    if tool_name == "calculator":
        expression = arguments.get("expression", "")

        if not expression:
            raise HTTPException(
                status_code=400,
                detail="calculator requires expression."
            )

        result = safe_calculate(expression)

        return {
            "tool": "calculator",
            "expression": expression,
            "result": result
        }

    if tool_name == "word_counter":
        text = arguments.get("text", "")
        words = text.split()

        return {
            "tool": "word_counter",
            "text": text,
            "word_count": len(words),
            "char_count": len(text)
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unknown tool: {tool_name}"
    )


def safe_calculate(expression: str):
    """
    Safer calculator.
    Do not use eval().
    Only supports numbers and basic math operators.
    """

    if len(expression) > 100:
        raise HTTPException(
            status_code=400,
            detail="Expression is too long."
        )

    try:
        tree = ast.parse(expression, mode="eval")
        result = eval_ast_node(tree.body)

        if abs(result) > 1_000_000_000:
            raise HTTPException(
                status_code=400,
                detail="Calculation result is too large."
            )

        return result

    except ZeroDivisionError:
        raise HTTPException(
            status_code=400,
            detail="Division by zero is not allowed."
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid math expression: {str(e)}"
        )


def eval_ast_node(node):
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        left = eval_ast_node(node.left)
        right = eval_ast_node(node.right)
        op_type = type(node.op)

        if op_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")

        return ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = eval_ast_node(node.operand)
        op_type = type(node.op)

        if op_type not in ALLOWED_OPERATORS:
            raise ValueError("Unary operator not allowed.")

        return ALLOWED_OPERATORS[op_type](operand)

    raise ValueError("Invalid expression.")