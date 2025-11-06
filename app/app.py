import torch
import os
import io
import gc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from unsloth import FastVisionModel
from PIL import Image
from datetime import datetime
from starlette_prometheus import PrometheusMiddleware, metrics # For monitoring

# --- Global Configuration ---
MODEL_PATH = "outputs/merged_gemma3_4b_it_lora" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- App Definition ---
app = FastAPI(title="Tetris Image Taging")

# Add Prometheus middleware for /metrics endpoint
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

# Global variables to hold the loaded model and processor
model = None
processor = None

# --- Model Loading (on Startup) ---
def load_vlm_model():
    """
    Loads the merged Unsloth VLM model once on server startup.
    """
    global model, processor
    if model is None:
        try:
            print(f"--- Loading model from {MODEL_PATH} ---")
            # Load merged model using bfloat16 for max performance (RTX 4090)
            # The processor is loaded internally by FastVisionModel.from_pretrained
            model, processor = FastVisionModel.from_pretrained(
                MODEL_PATH,
                max_seq_length = 2048,
                dtype = torch.bfloat16, # Best for 4090
                load_in_4bit = False,   # 4-bit is slower on 4090
                device_map = "auto",    # Use the GPU
                trust_remote_code=True,
            )
            
            # Apply Unsloth's inference optimizations
            FastVisionModel.for_inference(model)
            model.eval() # Set model to evaluation mode (disables dropout)
            
            print("--- VLM Model and Processor loaded successfully ---")
            
        except Exception as e:
            print(f"!!! FATAL ERROR: Failed to load VLM model: {e}")
            raise RuntimeError(f"Failed to load VLM model: {e}")

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler. Calls the model loader.
    """
    try:
        load_vlm_model()
    except RuntimeError:
        # If model fails to load, health check will report model_loaded: False
        pass

# --- Synchronous Inference Helper Function ---
# This heavy, blocking code runs in a separate threadpool
def run_vlm_inference(image: Image.Image, user_prompt: str):
    """
    Runs the actual model inference (synchronous blocking function).
    Must be called using run_in_threadpool.
    """
    if model is None or processor is None:
        raise RuntimeError("Model or processor is not loaded.")

    try:
        # --- Prompt Engineering ---
        if not user_prompt:
            prompt_text = """<image>

**دستورالعمل:**
تصویر ورودی یک محصول پوشاک است. آن را بررسی کن و فقط در قالب فارسی زیر، تگ‌های مرتبط را استخراج کن.
هیچ متن اضافه‌ای ننویس.
تمرکز بر آیتم اصلیِ پوشاک -
خروجی «کاملاً فارسی» باشد؛ اعداد و ویرگول فارسی (،) و بدون هیچ متن اضافه -
- جنس پارچه فقط اگر در تصویر قابل تشخیص است (نمونه‌ها: پنبه، کتان، پشم، پلی‌استر، جین/دنیم، ویسکوز، ابریشم، چرم/مصنوعی، بافتنی، مخمل، ریون، الاستین/اسپندکس). در غیر این صورت چیزی نگو.
- برند را فقط در صورت رویت «روشن و بدون تردید» (مثلاً برچسب/لوگو واضح) بنویس؛ وگرنه چیزی نگو.
- جنسیت را بر اساس طراحی/الگوی رایج در صورت اطمینان (زنانه/مردانه/بچه‌گانه/یونیسکس)؛ وگرنه چیزی نگو.
- دسته را «یک واژه از فهرست زیر» انتخاب کن: تی‌شرت، پیراهن، شلوار، شلوارک، دامن، مانتو، هودی، سویشرت، ژاکت/پلور، کاپشن/کت، بلوز/تونیک، شال/روسری، لباس‌خواب، لباس‌زیر، کفش، کیف، اکسسوری.
- تگ‌ها دقیق و کوتاه: ۵ کلیدواژه مرتبط با نوع، فصل، الگو/طرح (مثلاً چهارخانه، راه‌راه، ساده)، استایل (کژوال، رسمی، مجلسی، اسپرت، مینیمال)، فیت/قد/یقه/آستین، کاربرد (روزمره، مهمانی، زمستانی...). آن‌ها را با «،» جدا کن.


عنوان: [عنوان کوتاه محصول]
دسته: [دسته بندی پوشاک]
رنگ: [رنگ‌های اصلی]
جنس: [جنس پارچه (اگر مشخص است)]
تگ‌ها: [۵ تگ کلیدی مرتبط، جدا شده با کاما]

برای مثال:
عنوان: [پلور]
دسته: [لباس]
رنگ: [خاکستری]
جنس: [مخملی]
برند: [کارترز]
جنسیت:[پسرانه]
تگ‌ها: [ونگوگ، پاییزی، سایز متوسط]

درنهایت خروجی فقط مقادیر کلید ها باشند و خود عنوان کلید در خروجی نیار.
مثال خروجی نهایی:
پلیور، لباس، خاکستری، مخملی، کارترز، پسرانه، ونگوگ، پاییزی، سایز متوسط


عنوان:"""
        else:
            prompt_text = user_prompt

        # --- Set message order (Image first) ---
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}
        ]

        # --- Prepare Inputs ---
        chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        batch = processor(text=chat, images=image, return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items() if hasattr(v, "to")}

        # --- Clear CUDA Cache (Good practice for servers) ---
        torch.cuda.empty_cache()
        gc.collect()

        # --- Run Inference ---
        with torch.inference_mode():
            out = model.generate(
                **batch, 
                max_new_tokens=256, # Increased token limit
                do_sample=False, 
                use_cache=True
            )

        # --- Output Parsing ---
        full_result = processor.batch_decode(out, skip_special_tokens=True)[0]
        
        # Extract only the model's response, not the input prompt
        try:
            prediction = full_result.split("model\n")[-1]
            
            # Remove the echoed prompt from the output
            if prompt_text in prediction:
                 prediction = prediction.split(prompt_text, 1)[-1]
            
            prediction = prediction.strip()
            
            if not prediction and full_result:
                prediction = full_result # Fallback if parsing fails
                
        except Exception:
            prediction = full_result # On error, return the full text

        return {"prediction": prediction}

    except Exception as e:
        print(f"!!! Error during model inference: {e}")
        # Re-raise the exception to be caught by the main endpoint
        raise e

# --- API Endpoints ---

@app.post("/predict/")
async def predict(file: UploadFile = File(...), prompt: str = Form(default="")):
    """
    Image Taging 
    Accepts an image and a text prompt.
    """
    print("="*50)
    print(f"📥 Request received!")
    print(f"📄 Filename: {file.filename}")
    print(f"📝 Prompt length: {len(prompt)}")
    print(f"⏰ Time: {datetime.now()}")
    print("="*50)
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load on startup.")

    # --- Async Read ---
    try:
        contents = await file.read() # Async read of file
        image = Image.open(io.BytesIO(contents)).convert("RGB") # Sync, but fast
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

    # --- Async Execution in Threadpool ---
    # Run the blocking 'run_vlm_inference' function in a separate thread
    # This keeps the main FastAPI event loop free.
    try:
        result = await run_in_threadpool(run_vlm_inference, image, user_prompt=prompt)
        return result
    except Exception as e:
        # Catch errors from the inference function
        raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)}")

@app.get("/")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "OK", "model_loaded": model is not None}
