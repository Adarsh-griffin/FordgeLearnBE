import os
import requests
import json
import boto3
from groq import Groq
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from urllib.parse import urlparse
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()

# --- CONFIGURATION ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/images"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://jamdadeaditya05_db_user:Aditya6768@cluster0.2cdq0nw.mongodb.net/")
DB_NAME = "LearnForge"
COLLECTION_NAME = "files"

# AWS Config
AWS_AUDIO_BUCKET = os.getenv("AWS_AUDIO_BUCKET") or os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION") or "us-east-1"
AWS_IMAGE_PREFIX = "learning-images"

# Rate Limit: 4 calls per 1 second for Serper
CALLS = 4
PERIOD = 1

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def get_groq_client():
    api_key_str = os.getenv("GROQ_API_KEY", "")
    api_keys = [k.strip() for k in api_key_str.split(',') if k.strip()]
    
    if not api_keys:
        raise ValueError("GROQ_API_KEY is not set")
        
    return Groq(api_key=api_keys[0])

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def get_educational_images(query):
    """
    Fetches images from Serper.dev with a strict rate limit.
    """
    payload = json.dumps({
        "q": f"{query} educational diagram",
        "num": 10,  # Get 10 images to have more candidates
        "autocorrect": True,
        "safe": "active"
    })
    
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(SERPER_URL, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        images = [img['imageUrl'] for img in results.get('images', [])]
        return images
    except Exception as e:
        print(f"Error fetching images for '{query}': {e}")
        return []

def generate_keywords_from_summary(summary_text):
    """
    Uses Groq to extract search keywords from the summary.
    """
    client = get_groq_client()
    
    prompt = f"""
    Analyze the following educational summary and extract 4 distinct, visual search keywords 
    that would yield good educational diagrams or illustrations.
    Return ONLY a JSON array of strings. No other text.
    
    Summary:
    {summary_text[:2000]}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts visual keywords."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        content = chat_completion.choices[0].message.content
        # Clean up code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        keywords = json.loads(content)
        return keywords if isinstance(keywords, list) else []
    except Exception as e:
        print(f"Error generating keywords: {e}")
        return []

def upload_image_to_s3(image_url, folder="default", target_filename=None):
    """
    Downloads image from URL, converts to PNG, and uploads to S3.
    Returns the public S3 URL on success, None on failure.
    """
    if not AWS_AUDIO_BUCKET:
        print("  ⚠️  AWS_AUDIO_BUCKET not set, skipping upload")
        return None

    try:
        # Download image with timeout
        print(f"  📥 Downloading: {image_url[:80]}...")
        resp = requests.get(image_url, stream=True, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        resp.raise_for_status()
        
        # Check content type (but allow empty/missing for some sources)
        content_type = resp.headers.get('content-type', '').lower()
        
        # Some image CDNs don't send proper Content-Type, so we'll try to validate with PIL
        if content_type and not content_type.startswith('image/'):
            # Only reject if we have a Content-Type and it's definitely not an image
            if content_type.startswith(('text/', 'application/json', 'application/xml')):
                print(f"  ❌ Invalid content type: {content_type}")
                return None
        
        # Determine filename
        if target_filename:
            s3_key = f"{AWS_IMAGE_PREFIX}/{folder}/{target_filename}"
        else:
            parsed = urlparse(image_url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"image_{int(os.urandom(4).hex(), 16)}.png"
            safe_name = secure_filename(filename)
            s3_key = f"{AWS_IMAGE_PREFIX}/{folder}/{safe_name}"

        # Convert to PNG using Pillow (also validates it's a real image)
        try:
            image_data = BytesIO(resp.content)
            img = Image.open(image_data)
            img.verify()  # Verify it's a valid image
            
            # Re-open for processing (verify() closes the image)
            image_data.seek(0)
            img = Image.open(image_data)
        except Exception as e:
            print(f"  ❌ Invalid image file: {str(e)[:60]}")
            return None
        
        # Convert RGBA to RGB if saving with transparency issues
        if img.mode in ('RGBA', 'LA', 'P'):
            # Keep transparency for PNG
            pass
        
        out_img = BytesIO()
        img.save(out_img, format='PNG')
        out_img.seek(0)
        
        # Upload to S3
        print(f"  ☁️  Uploading to S3: {s3_key}")
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3.upload_fileobj(
            out_img,
            AWS_AUDIO_BUCKET,
            s3_key,
            ExtraArgs={
                'ContentType': 'image/png'
                # Note: ACL not set - bucket uses bucket policy for public access
            }
        )
        
        # Construct public URL
        if AWS_REGION == 'us-east-1':
            s3_url = f"https://{AWS_AUDIO_BUCKET}.s3.amazonaws.com/{s3_key}"
        else:
            s3_url = f"https://{AWS_AUDIO_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        
        print(f"  ✅ Uploaded successfully")
        return s3_url
        
    except requests.exceptions.Timeout:
        print(f"  ❌ Timeout downloading image from {image_url[:60]}...")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"  ❌ HTTP error {e.response.status_code} downloading image")
        return None
    except Exception as e:
        print(f"  ❌ Failed to upload {image_url[:60]}... to S3: {str(e)[:100]}")
        return None

def process_images_for_file(file_path, summary_text):
    """
    Full pipeline: Summary -> Keywords -> Images -> S3 -> MongoDB
    """
    if not summary_text:
        print("No summary text provided.")
        return []
        
    print(f"\n🖼️ Starting Image Finder for {file_path}...")
    
    # 1. Generate Keywords
    print("Step 1: Generating keywords...")
    keywords = generate_keywords_from_summary(summary_text)
    print(f"🔑 Keywords: {keywords}")
    
    if not keywords:
        print("No keywords generated.")
        return []

    all_images = []
    
    # 2. Find Images
    print("Step 2: Finding images...")
    for kw in keywords:
        imgs = get_educational_images(kw)
        if imgs:
            # Collect ALL images from each keyword (up to 10)
            all_images.extend(imgs)
            
    # Deduplicate
    unique_images = list(dict.fromkeys(all_images))
    print(f"🔍 Found {len(unique_images)} unique image candidates.")
    
    # 3. Upload to S3 - Try all images but keep only first 4 successful
    print("Step 3: Uploading to S3 (trying all, keeping best 4)...")
    final_urls = []
    
    # Clean up base file name for the folder and the image prefix
    base_filename_full = os.path.basename(file_path)
    base_name_clean = os.path.splitext(base_filename_full)[0]
    safe_folder = secure_filename(base_name_clean)
    
    image_counter = 1
    for i, img_url in enumerate(unique_images):
        # Stop once we have 4 successful uploads
        if len(final_urls) >= 4:
            print(f"\n✅ Successfully uploaded 4 images, stopping.")
            break
            
        print(f"\n🖼️  Trying candidate {i+1}/{len(unique_images)}...")
        target_name = f"{safe_folder}-learning-image-{image_counter}.png"
        
        s3_url = upload_image_to_s3(img_url, folder=safe_folder, target_filename=target_name)
        
        if s3_url:
            final_urls.append(s3_url)
            print(f"✅ Image {image_counter} uploaded: {s3_url}")
            image_counter += 1
        else:
            print(f"⚠️  Candidate {i+1} failed, trying next...")
        
    # 4. Update MongoDB
    print("Step 4: Updating MongoDB...")
    try:
        col = get_mongo_collection()
        res = col.update_one(
            {"filePath": file_path},
            {"$set": {"images": final_urls}}
        )
        
        if res.modified_count > 0:
            print(f"✅ Successfully saved {len(final_urls)} images to MongoDB for {file_path}")
        else:
            res = col.update_one(
                {"filename": base_filename_full},
                {"$set": {"images": final_urls}}
            )
            if res.modified_count > 0:
                 print(f"✅ Successfully saved {len(final_urls)} images to MongoDB for {base_filename_full} (by filename)")
            else:
                 print(f"ℹ️ Document matched but not modified (images might be same) or not found.")
                 
    except Exception as e:
        print(f"Error updating MongoDB: {e}")
            
    return final_urls

if __name__ == "__main__":
    # Test run
    test_summary = "Photosynthesis is the process used by plants to convert light energy into chemical energy."
    print("Testing pipeline with dummy summary...")
    # Mocking file path
    process_images_for_file("test_photosynthesis.pdf", test_summary)