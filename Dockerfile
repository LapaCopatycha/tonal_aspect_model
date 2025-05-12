FROM python:3.12-slim

WORKDIR /app

COPY .. /app/

EXPOSE 80

RUN pip install --upgrade virtualenv
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r runtime-requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]