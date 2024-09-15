import uvicorn
from fastapi import FastAPI, Query
from models.text import load_text_model, generate_text


app = FastAPI()

@app.get("/generate/text")
def serve_language_model_controller(prompt=Query(...)):
	pipeline = load_text_model()
	output = generate_text(pipeline, prompt)
	return output


if __name__ == "__main__":
	uvicorn.run("app:app", port=8000, reload=False)
