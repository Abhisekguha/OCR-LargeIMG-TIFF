import gradio as gr
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
from PIL import Image
from jiwer import wer, cer
import io
import csv
import matplotlib.pyplot as plt

# Initialize PaddleOCR 
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def process_large_image(img, tile_height=2500, tile_width=3000, overlap=50):              
    h, w = img.shape[:2]
    processed_results = []

    for y in range(0, h, tile_height - overlap):
        for x in range(0, w, tile_width - overlap):
            # Create a tile from the image
            tile = img[y:y+tile_height, x:x+tile_width]
            # Process the tile with PaddleOCR
            result = ocr.ocr(tile)
            processed_results.append((result, (y, x)))

    return processed_results

def normalize_text(text):
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def save_results_to_csv(numbers, texts):
    """
    Save extracted numbers and texts to a CSV file
    """
    filename = "ocr_extraction_results.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Type', 'Value', 'Coordinates'])
        
        # Write numbers
        for n in numbers:
            csvwriter.writerow(['Number', n['number'], str(n['bbox'])])
        
        # Write texts
        for t in texts:
            csvwriter.writerow(['Text', t['text'], str(t['bbox'])])
    
    return filename

def plot_error_rates(wer_score, cer_score):
    """
    Create a bar plot for WER and CER
    """
    plt.figure(figsize=(8, 5))
    plt.bar(['Word Error Rate (WER)', 'Character Error Rate (CER)'], 
            [wer_score, cer_score], 
            color=['blue', 'green'])
    plt.title('OCR Error Rates')
    plt.ylabel('Error Rate')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Add value labels on top of each bar
    plt.text(0, wer_score, f'{wer_score:.4f}', 
             ha='center', va='bottom')
    plt.text(1, cer_score, f'{cer_score:.4f}', 
             ha='center', va='bottom')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('error_rates.png')
    plt.close()
    
    return 'error_rates.png'

def process_ocr(image, ground_truth=None):
    # Check if image is None
    if image is None:
        return (
            "No image uploaded.", 
            None, 
            [], 
            None, 
            "No text extracted", 
            "No text extracted", 
            None,
            None
        )

    # Convert image to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        img_cv = np.array(image)
    else:
        img_cv = image

    # Check if the image is in RGB mode
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_cv

    # Create a copy for annotation
    annotated_img = img_cv.copy()

    # Process large image by tiling
    result = process_large_image(img_rgb)

    # Initialize tracking variables with safe default values
    total_confidence = 0
    num_words = 0
    accuracy = 0.0

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
                    if confidence < 0.05:
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
    accuracy = total_confidence / num_words if num_words > 0 else 0.0

    # Prepare extracted text
    detected_texts = [t['text'] for t in text_data]
    cleaned_detected_text = normalize_text(" ".join(detected_texts))

    extracted_numbers = "\n".join([f"Number: {n['number']}, Bounding Box: {n['bbox']}" for n in numbers]) or "No numbers extracted"
    extracted_texts = "\n".join([f"Text: {t['text']}, Bounding Box: {t['bbox']}" for t in text_data]) or "No texts extracted"
    
    # Compute WER and CER if ground truth is provided
    wer_score = None
    cer_score = None
    error_plot = None
    if ground_truth:
        cleaned_ground_truth = normalize_text(ground_truth)
        wer_score = wer(cleaned_ground_truth, cleaned_detected_text)
        cer_score = cer(cleaned_ground_truth, cleaned_detected_text)
        error_plot = plot_error_rates(wer_score, cer_score)

    # Convert annotated image to PIL for Gradio
    pil_annotated_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    
    # Save results to CSV
    csv_filename = save_results_to_csv(numbers, text_data)
    
    return (
        f"Average Accuracy: {accuracy:.2f}",
        pil_annotated_img,
        csv_filename,
        extracted_numbers,
        extracted_texts,
        error_plot,
        {
            "Word Error Rate (WER)": wer_score,
            "Character Error Rate (CER)": cer_score
        } if ground_truth else None
    )

def launch_ocr_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Large Image OCR Processing")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Left Column: Input Components
                image_input = gr.Image(type="pil", label="Upload Large Image (TIFF/PNG/JPG)", height=400)
                ground_truth_input = gr.Textbox(label="Ground Truth Text (Optional)", lines=3)
                process_btn = gr.Button("Process Image", variant="primary")
            
            with gr.Column(scale=1):
                # Right Column: Annotated Image
                accuracy_output = gr.Textbox(label="Accuracy")
                annotated_image_output = gr.Image(label="Annotated Image with Recognized Texts", height=600)
                
        
        with gr.Row():
            with gr.Column(scale=1):
                # Extracted Numbers and Texts with CSV Download
                csv_output = gr.File(label="Download OCR Results CSV")
                numbers_output = gr.Textbox(label="Extracted Numbers", lines=3)
                texts_output = gr.Textbox(label="Extracted Texts", lines=3)
                
                # Error Rate Plot
                error_plot_output = gr.Image(label="WER and CER Error Rates")
        
        # Processing Workflow
        process_btn.click(
            fn=process_ocr, 
            inputs=[image_input, ground_truth_input], 
            outputs=[
                accuracy_output, 
                annotated_image_output, 
                csv_output,
                numbers_output, 
                texts_output, 
                error_plot_output,
                gr.JSON(label="Error Metrics (WER/CER)")
            ]
        )

    return demo

# Launch the app
if __name__ == "__main__":
    demo = launch_ocr_app()
    demo.launch(server_name="127.0.0.1", server_port=7860)