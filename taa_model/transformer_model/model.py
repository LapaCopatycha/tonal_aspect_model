from transformers import AutoTokenizer, AutoModelForSequenceClassification
from configurations import BASE_DIR


model_path = f'{BASE_DIR}/taa_model/model_history/transformers/saved_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)