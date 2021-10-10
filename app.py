from fastapi import FastAPI
import classify
app = FastAPI()

@app.get("/category/{sentence}")
def get_category(sentence):
    return classify.get_category(sentence)