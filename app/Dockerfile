FROM unsloth/unsloth

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./outputs/merged_gemma3_4b_it_lora /app/outputs/merged_gemma3_4b_it_lora

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]