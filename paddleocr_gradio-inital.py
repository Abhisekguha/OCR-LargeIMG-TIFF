import gradio as gr
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
from PIL import Image
from jiwer import wer, cer
import io

# Check if Paddle is using GPU
print(f"Using GPU: {paddle.is_compiled_with_cuda()}")

# Initialize PaddleOCR 
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)



# Function to preprocess the image
def preprocess_image(img, contrast_alpha=2.0, contrast_beta=70):   # prev alpha-2.0, beta-50
    # Convert to grayscale for better contrast
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    enhanced_img = cv2.convertScaleAbs(gray_img, alpha=contrast_alpha, beta=contrast_beta)
    
    # Apply denoising (GaussianBlur)
    denoised_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    
    return denoised_img

# def adaptive_overlap_strategy(img, tile_height, tile_width):
#     """
#     Dynamically calculate optimal overlap based on image characteristics
    
#     Parameters:
#     - img: Input image
#     - tile_height: Height of processing tiles
#     - tile_width: Width of processing tiles
    
#     Returns:
#     - width_overlap: Horizontal overlap
#     - height_overlap: Vertical overlap
#     """
#     # Image dimensions
#     height, width = img.shape[:2]
    
#     # Basic overlap calculation
#     base_width_overlap = max(50, int(width * 0.02))    # 1% of width or min 50 pixels
#     base_height_overlap = max(50, int(height * 0.02))  # 1% of height or min 50 pixels
    
#     # Analyze image complexity
#     # Convert to grayscale if needed
#     if len(img.shape) > 2:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img
    
#     # Calculate image complexity using Laplacian variance
#     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
#     # Complexity-based overlap adjustment
#     complexity_factor = np.log1p(laplacian_var) / 10  # Logarithmic scaling
    
#     # Dynamically adjust overlap
#     width_overlap = int(base_width_overlap * (1 + complexity_factor))
#     height_overlap = int(base_height_overlap * (1 + complexity_factor))
    
#     # Ensure overlap doesn't exceed tile dimensions
#     width_overlap = min(width_overlap, tile_width // 2)
#     height_overlap = min(height_overlap, tile_height // 2)
    
#     return width_overlap, height_overlap

# def process_large_image(img, tile_height=2500, tile_width=3000, overlap_strategy='adaptive'):
#     """
#     Process large images with adaptive or fixed overlap
    
#     Parameters:
#     - img: Input image
#     - tile_height: Height of processing tiles
#     - tile_width: Width of processing tiles
#     - overlap_strategy: 'adaptive' or a fixed overlap value
    
#     Returns:
#     - List of processed tile results
#     """
#     if overlap_strategy == 'adaptive':
#         # Dynamically calculate overlap
#         width_overlap, height_overlap = adaptive_overlap_strategy(img, tile_height, tile_width)
#     elif isinstance(overlap_strategy, (int, float)):
#         # Use fixed overlap
#         width_overlap = height_overlap = overlap_strategy
#     else:
#         # Default to a standard overlap
#         width_overlap = height_overlap = 50
    
#     # Debug print for overlap information
#     print(f"Processing with Width Overlap: {width_overlap}, Height Overlap: {height_overlap}")
    
#     processed_results = []
#     for y in range(0, img.shape[0], tile_height - height_overlap):
#         for x in range(0, img.shape[1], tile_width - width_overlap):
#             # Ensure tile doesn't exceed image dimensions
#             tile = img[y:min(y+tile_height, img.shape[0]), 
#                        x:min(x+tile_width, img.shape[1])]
            
#             # Process tile with PaddleOCR
#             result = ocr.ocr(tile)
#             processed_results.append((result, (y, x)))
    
#     return processed_results

def process_large_image(img, tile_height=2500, tile_width=3000, overlap=50):              # 2500, 3000, 50 
    h, w = img.shape[:2]  # Only get height and width (ignores channels if grayscale)
    processed_results = []

    for y in range(0, h, tile_height - overlap):
        for x in range(0, w, tile_width - overlap):
            # Create a tile from the image
            tile = img[y:y+tile_height, x:x+tile_width]
            # Process the tile with PaddleOCR
            result = ocr.ocr(tile)
            processed_results.append((result, (y, x)))  # Store result with tile position

    return processed_results

import re


# More lenient normalization
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Keep more characters
    # text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# def normalize_text(text):
#     """
#     Normalize text by removing special characters, extra spaces, and converting to lowercase.
#     """
    
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
#     return text

# Main OCR processing function for Gradio
def process_ocr(image, ground_truth=None):
    # Check if image is None
    if image is None:
        return "No image uploaded.", None, None, None, None

    # Convert image to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        img_cv = np.array(image)
    else:
        img_cv = image

    # Check if the image is in RGB mode
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:  # If it has color channels
        # Convert BGR image to RGB for PaddleOCR
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    else:
        # If the image has no color (grayscale), just use it as-is
        img_rgb = img_cv

    # Create a copy for annotation
    annotated_img = img_cv.copy()

    # Process large image by tiling
    result = process_large_image(img_rgb)

    total_confidence = 0
    num_words = 0

    # Initialize lists for extracted data
    numbers = []
    text_data = []

    # Loop through the results
    for tile_result, (y_offset, x_offset) in result:
        for line in tile_result:
            if line:
                for word_info in line:
                    text = word_info[-1][0]  # Extract detected text
                    confidence = word_info[-1][1]  # Extract confidence score

                    # Filter out low-confidence detections
                    if confidence < 0.05:        #0.3
                        continue

                    total_confidence += confidence
                    num_words += 1

                    bbox = word_info[0]
                    # Adjust bounding box for the tile offset
                    bbox = [(int(point[0] + x_offset), int(point[1] + y_offset)) for point in bbox]

                    # Draw bounding boxes on the annotated image
                    cv2.polylines(annotated_img, [np.array(bbox)], True, (0, 0, 255), 2)
                    # Put the detected text on the image
                    cv2.putText(annotated_img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    # Check if the detected text is a number
                    if text.replace('.', '', 1).isdigit():
                        numbers.append({"number": text, "bbox": bbox})
                    else:
                        text_data.append({"text": text, "bbox": bbox})

    # Calculate average accuracy
    accuracy = total_confidence / num_words if num_words > 0 else 0

    # Prepare extracted text
    detected_texts = [t['text'] for t in text_data]
    cleaned_detected_text = normalize_text(" ".join(detected_texts))

    extracted_numbers = "\n".join([f"Number: {n['number']}, Bounding Box: {n['bbox']}" for n in numbers])
    extracted_texts = "\n".join([f"Text: {t['text']}, Bounding Box: {t['bbox']}" for t in text_data])
    
    # Compute WER and CER if ground truth is provided
    wer_score = None
    cer_score = None
    if ground_truth:
        cleaned_ground_truth = normalize_text(ground_truth)
        wer_score = wer(cleaned_ground_truth, cleaned_detected_text)
        cer_score = cer(cleaned_ground_truth, cleaned_detected_text)

    # Convert annotated image to PIL for Gradio
    pil_annotated_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    
    return (
        f"Average Accuracy: {accuracy:.2f}",
        pil_annotated_img,
        extracted_numbers,
        extracted_texts,
        {
            "Word Error Rate (WER)": wer_score,
            "Character Error Rate (CER)": cer_score
        } if ground_truth else None
    )

# Create Gradio interface
def launch_ocr_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Large Image OCR Processing with PaddleOCR")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Large Image (TIFF/PNG/JPG)")
                ground_truth_input = gr.Textbox(label="Ground Truth Text (Optional)", lines=3)
                process_btn = gr.Button("Process Image")
        
        with gr.Row():
            with gr.Column():
                accuracy_output = gr.Textbox(label="Accuracy")
                annotated_image_output = gr.Image(label="Annotated Image")
                
        with gr.Row():
            with gr.Column():
                numbers_output = gr.Textbox(label="Extracted Numbers")
                texts_output = gr.Textbox(label="Extracted Texts")
                error_metrics_output = gr.JSON(label="Error Metrics (WER/CER)")
        
        # Set up the processing workflow
        process_btn.click(
            fn=process_ocr, 
            inputs=[image_input, ground_truth_input], 
            outputs=[
                accuracy_output, 
                annotated_image_output, 
                numbers_output, 
                texts_output, 
                error_metrics_output
            ]
        )

    return demo

# Launch the app
if __name__ == "__main__":
    demo = launch_ocr_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
