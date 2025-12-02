from PIL import Image
from models.ocr import OCR
from config import VOCAB_SIZE
from utils.visualization import visualize


def main():
    """Main entry point for the OCR application."""
    # Initialize OCR model
    ocr_model = OCR(vocab_size=VOCAB_SIZE)
    
    # Image path
    image_path = "new_data/train_val_images/train_images/000a533ef1b9cacf.jpg"
    
    # Run OCR
    pred_text, bboxes, (new_width, new_height) = ocr_model(
        image_path=image_path, 
        enhance_threshold=0.94, 
        threshold=0.85
    )
    
    # Print results
    print(f"Bounding Boxes: {bboxes}")
    print(f"Predicted Text: {pred_text}")
    
    # Visualize results
    image = Image.open(image_path).convert('RGB')
    visualize(image, bboxes, pred_text, new_width, new_height)


if __name__ == "__main__":
    main()
