import requests
import json
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import keys

class ImageGenerator:
    """Handles image generation for OpenAI and Gemini models"""
    
    def __init__(self):
        self.openai_api_key = keys.OPENAI_API_KEY
        self.google_api_key = keys.GOOGLE_API_KEY
        genai.configure(api_key=self.google_api_key)
    
    def generate_openai_image(self, prompt, size="1024x1024", quality="high", n=1):
        """Generate image using OpenAI's gpt-image-1 model"""
        url = "https://api.openai.com/v1/images/generations"
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract base64 image data
            if result.get('data') and len(result['data']) > 0:
                b64_json = result['data'][0]['b64_json']
                
                # Convert base64 to PIL Image
                image_data = base64.b64decode(b64_json)
                image = Image.open(BytesIO(image_data))
                
                return image, None
            else:
                return None, "No image data received from OpenAI"
                
        except requests.exceptions.RequestException as e:
            return None, f"OpenAI API request failed: {str(e)}"
        except Exception as e:
            return None, f"Error generating OpenAI image: {str(e)}"
    
    def generate_gemini_image(self, prompt, orientation="SQUARE", style="PHOTOGRAPHY"):
        """Generate image using Google's Imagen 3 model via predict method"""
        try:
            # Use the corrected predict endpoint approach
            url = "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.google_api_key
            }
            
            # Use the corrected instances structure
            data = {
                "instances": [
                    {
                        "prompt": prompt
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Process the response from predictions
            if result.get('predictions') and len(result['predictions']) > 0:
                prediction = result['predictions'][0]
                
                # The image data is directly in the prediction as Base64 encoded
                if 'bytesBase64Encoded' in prediction:
                    b64_string = prediction['bytesBase64Encoded']
                    
                    # Decode the Base64 string
                    image_bytes = base64.b64decode(b64_string)
                    image = Image.open(BytesIO(image_bytes))
                    return image, None
                else:
                    return None, "No bytesBase64Encoded data in prediction response"
            else:
                return None, "No predictions returned from Gemini"
                
        except requests.exceptions.RequestException as e:
            return None, f"Gemini API request failed: {str(e)}"
        except Exception as e:
            return None, f"Error generating Gemini image: {str(e)}"
    
    def generate_image(self, model_name, prompt, **kwargs):
        """Generate image based on model name"""
        if model_name == "gpt-image-1":
            return self.generate_openai_image(prompt, **kwargs)
        elif model_name == "imagen-3.0-generate-002":
            return self.generate_gemini_image(prompt, **kwargs)
        else:
            return None, f"Unsupported image generation model: {model_name}"
    
    def save_image_to_temp(self, image, filename_prefix="generated_image"):
        """Save PIL Image to a temporary file and return the path"""
        import tempfile
        import os
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        filename = f"{filename_prefix}_{hash(str(image.tobytes()))}.png"
        temp_path = os.path.join(temp_dir, filename)
        
        # Save the image
        image.save(temp_path, "PNG")
        
        return temp_path