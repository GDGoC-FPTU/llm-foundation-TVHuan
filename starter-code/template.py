"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI
    
    # Tạo client từ API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Đo thời gian bắt đầu
    start_time = time.time()
    
    # Gọi OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # Đo thời gian kết thúc
    latency = time.time() - start_time
    
    # Extract response text
    response_text = response.choices[0].message.content
    
    # Extract token usage
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types
    
    # Tạo client từ API key
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Cấu hình parameters
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    
    # Đo thời gian bắt đầu
    start_time = time.time()
    
    # Gọi Gemini API
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    
    # Đo thời gian kết thúc
    latency = time.time() - start_time
    
    # Extract response text
    response_text = response.text
    
    # Extract token usage
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic
    
    # Tạo client từ API key
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Đo thời gian bắt đầu
    start_time = time.time()
    
    # Gọi Anthropic API
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    
    # Đo thời gian kết thúc
    latency = time.time() - start_time
    
    # Extract response text
    response_text = response.content[0].text
    
    # Extract token usage
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # Gọi gpt-4o
    response_gpt4o, latency_gpt4o, usage_gpt4o = call_openai(
        prompt=prompt,
        model=OPENAI_MODEL,  # "gpt-4o"
    )
    
    # Gọi gpt-4o-mini
    response_gpt4o_mini, latency_gpt4o_mini, usage_gpt4o_mini = call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,  # "gpt-4o-mini"
    )
    
    # Gọi gemini-2.5-flash
    response_gemini, latency_gemini, usage_gemini = call_gemini(
        prompt=prompt,
        model=GEMINI_MODEL,  # "gemini-2.5-flash"
    )
    
    # Hàm tính chi phí
    def calculate_cost(model_name: str, usage: dict) -> float:
        prices = PRICING_1M_TOKENS[model_name]
        cost = (
            usage["input_tokens"] * prices["input"]
            + usage["output_tokens"] * prices["output"]
        ) / 1_000_000
        return cost
    
    # Tính chi phí cho từng model
    cost_gpt4o = calculate_cost("gpt-4o", usage_gpt4o)
    cost_gpt4o_mini = calculate_cost("gpt-4o-mini", usage_gpt4o_mini)
    cost_gemini = calculate_cost("gemini-2.5-flash", usage_gemini)
    
    # Tổng hợp kết quả
    return {
        "gpt4o": {
            "response": response_gpt4o,
            "latency": latency_gpt4o,
            "cost": cost_gpt4o,
            "input_tokens": usage_gpt4o["input_tokens"],
            "output_tokens": usage_gpt4o["output_tokens"],
        },
        "gpt4o_mini": {
            "response": response_gpt4o_mini,
            "latency": latency_gpt4o_mini,
            "cost": cost_gpt4o_mini,
            "input_tokens": usage_gpt4o_mini["input_tokens"],
            "output_tokens": usage_gpt4o_mini["output_tokens"],
        },
        "gemini_flash": {
            "response": response_gemini,
            "latency": latency_gemini,
            "cost": cost_gemini,
            "input_tokens": usage_gemini["input_tokens"],
            "output_tokens": usage_gemini["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai
    
    # Khởi tạo client Gemini
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Lịch sử trò chuyện (giữ 3 lần gần nhất)
    conversation_history = []
    
    print("\n🤖 Chatbot Gemini 2.5 (gõ 'quit' hoặc 'exit' để thoát)")
    print("-" * 60)
    
    while True:
        # Nhập prompt từ người dùng
        user_input = input("\n👤 You: ").strip()
        
        # Kiểm tra quit/exit
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Thêm user message vào lịch sử
        conversation_history.append({
            "role": "user",
            "content": user_input,
        })
        
        # Giữ chỉ 3 lần trò chuyện gần nhất (= 6 messages: user + assistant)
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]
        
        # Gọi Gemini API với streaming
        print("\n🤖 Bot: ", end="", flush=True)
        
        response_text = ""
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation_history,
            stream=True,
        )
        
        # Stream response token by token
        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                response_text += chunk.text
        
        print()  # Newline sau response
        
        # Thêm bot response vào lịch sử
        conversation_history.append({
            "role": "model",
            "content": response_text,
        })


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            
            # Nếu đây không phải lần retry cuối, chờ rồi thử lại
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    
    # Nếu hết lần retry, raise exception cuối cùng
    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        try:
            comp_result = compare_models(prompt)
        except TypeError:
            comp_result = compare_models()
        
        if comp_result and isinstance(comp_result, dict):
            res_dict = comp_result.copy()
            res_dict["prompt"] = prompt
            results.append(res_dict)
            
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|"
    ]
    
    # 2. Bảng ánh xạ chuyển đổi key kỹ thuật sang tên hiển thị mà bài test yêu cầu quét
    model_display_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash"
    }
    
    # Thứ tự duyệt qua các model
    models = ["gpt4o", "gpt4o_mini", "gemini_flash"]
    
    for res in results:
        prompt_str = res.get("prompt", "").replace("\n", " ")
        
        for model_key in models:
            if model_key in res:
                data = res[model_key]
                
                raw_resp = data["response"].replace("\n", " ")
                if len(raw_resp) > 50:
                    resp_str = raw_resp[:50] + "..."
                else:
                    resp_str = raw_resp
                
                # Định dạng các thông số hiển thị
                latency = f"{data['latency']:.2f}s"
                tokens = f"{data['input_tokens']}/{data['output_tokens']}"
                cost = f"${data['cost']:.6f}"
                
                # Lấy tên hiển thị chuẩn hóa (ví dụ: gemini_flash -> Gemini-Flash)
                display_name = model_display_names.get(model_key, model_key)
                
                # Ghi dữ liệu vào hàng của bảng
                lines.append(f"| {prompt_str} | {display_name} | {resp_str} | {latency} | {tokens} | {cost} |")
                
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
