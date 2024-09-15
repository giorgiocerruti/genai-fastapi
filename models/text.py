import torch 
from transformers import Pipeline, pipeline

prompt = "How to setup a FastAPI project?"

system_prompt = """
Your name is FastAPI Bot and you are a helpful chatbot responsible for teaching FastAPI to your users.
Always respond in markdown 
"""


def load_text_model():
	pipe = pipeline(
		"text-generation",
		model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
		torch_dtype=torch.bfloat16,
	) #download TinyLlama into tourch float16, less oprecise than float32
	
	return pipe


def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
	message = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": prompt},
	]#prepar the message list, always start with a system prompt to help the LLM
	
	prompt = pipe.tokenizer.apply_chat_template(
		message, 
		tokenize=False, #ask to generate text not tokens 
		add_generation_prompt=True #force the model to generate text based on chat history
	)#convert a list of chat messages into token

	predictions = pipe(
		prompt, 
		temperature=temperature, # Controls the randomes of the generation
		max_new_tokens=256, # specify the maximum generation token number
		do_sample=True, # picjk random token from a list of suitables tokens
		top_k=50, #create a list of top 50 must suitable tokens to pick from the current token in predicion step.
		top_p=0.95, # nucleus sampling, when create alist of suitable token, create alist of top tokens util your are satisfied that your list has 95% of the most suitable tokens to pick from 
	)#pass the preperred prompt to the model with inference parameters

	output = predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1] #LLamaTiny responds with the whole chat history, pick only the last message wich starts with "assistant:"

	return output

if __name__ == "__main__":
	
	pipe = load_text_model()
	output = generate_text(pipe, prompt)
	print(output)
		
		
