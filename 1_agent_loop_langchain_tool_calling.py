from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from dotenv import load_dotenv

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

load_dotenv()  # Load environment variables from .env file

@tool
def get_product_price(product_name: str) -> int:
    """
    Look up the price of the product in the catalog.
    """
    # In a real implementation, this function would query a database or an API
    # to get the price of the product. Here, we will just return a mock price.
    mock_prices ={"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return mock_prices.get(product_name.lower(), "Product not found")


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""

    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)



@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    print("Running agent loop...",question)

    tools=[get_product_price, apply_discount]
    tools_dict={ele.name:ele for ele in tools}

    llm=init_chat_model(
        f"ollama:{MODEL}",
        temperature=0,
    )

    llm_with_tools=llm.bind_tools(tools)

    messages=[
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]


    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\nIteration {iteration}...")


        ai_message=llm_with_tools.invoke(messages)
        tool_calls=ai_message.tool_calls

        if not tool_calls:
            print(ai_message.content)
            break

        # print(f"Tool calls made: {len(tool_calls)}",ai_message)
        # Tool calls made: 1 content='' additional_kwargs={} response_metadata={'model': 'qwen3:1.7b', 'created_at': '2026-08-02T18:47:24.0914733Z', 'done': True, 'done_reason': 'stop', 'total_duration': 13357611000, 'load_duration': 282715900, 'prompt_eval_count': 345, 'prompt_eval_duration': 73641000, 'eval_count': 179, 'eval_duration': 12994451000, 'logprobs': None, 'model_name': 'qwen3:1.7b', 'model_provider': 'ollama'} id='lc_run--019fc3cd-2a8d-7df1-bad1-27ac3288187a-0' 

        # tool_calls=[
        # {'name': 'get_product_price', 
        # 'args': {'product_name': 'laptop'}, 
        # 'id': '0c23c30e-983e-4a34-875b-cfb2a6df3e59', 
        # 'type': 'tool_call'}
        # ] 
        # invalid_tool_calls=[] usage_metadata={'input_tokens': 345, 'output_tokens': 179, 'total_tokens': 524}

        print(f">>>Tool call made:{tool_calls}")
        for tool_call in tool_calls:

            tool_call_name=tool_call.get("name")
            tool_call_args=tool_call.get("args")
            tool_call_id=tool_call.get("id")

            if not tool_call_name or not tool_call_args:
                print("Invalid tool call:",tool_call)
                continue

            if not tool_call_name in tools_dict:
                print("Tool not found:",tool_call_name)
                continue

            tool_function=tools_dict.get(tool_call_name)
            observation=tool_function.invoke(tool_call_args)

            print(f"Tool call done:{observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation),tool_call_id=tool_call_id)
        )



if __name__ == "__main__":
    question = "What is the price of a laptop with a silver discount?"
    run_agent(question)