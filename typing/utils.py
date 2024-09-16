from dataclasses import dataclass
from typing import TypedDict, Literal
from loguru import logger
import tiktoken

SupportedModel = Literal["gpt-3.5", "gpt-4"]
PriceTable = dict[SupportedModel: float]
prices: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200}

@dataclass
class Message(TypedDict):
    prompt: str
    response: str | None
    model: SupportedModel
    price: float

@dataclass
class MessageCostReport(TypedDict):
    req_costs: float
    res_costs: float
    total_costs: float

def count_tokens(text: str | None) -> int:
    encoder = "cl100k_base"
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    
    if text is None:
        logger.warning("Text is empty")
        return 0
    
    enc = tiktoken.get_encoding(encoder)
    return len(enc.encode(text))

def calculate_usage_costs(message: Message) -> MessageCostReport:
    if message["model"] not in prices:
        logger.error(f"Model {message["model"]} is not supported")
        raise ValueError(f"Model {message['model']} is not supported")
    req_costs = prices[message["model"]]
    req_costs = round(message["price"] * count_tokens(message["prompt"]), 3)
    res_costs = message["price"] * count_tokens(message["response"])
    total_costs = round(req_costs + res_costs, 3)
    return MessageCostReport(req_costs=req_costs, res_costs=res_costs, total_costs=total_costs)


if __name__ == "__main__":
    message = Message(prompt=input("Enter prompt: "), response="Hi giorgio", model="gpt-3.5", price=0.0030)
    print(message)
    report = calculate_usage_costs(message)
    logger.info(report)