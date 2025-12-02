import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def visualize(image, bboxes, pred_text, new_width, new_height):
    # Resize the PIL image first
    image = image.resize((new_width, new_height), Image.BILINEAR)
    
    # Convert PIL Image to NumPy array (already in RGB format)
    image_np = np.array(image)
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 10))
    
    # Display image and bounding boxes
    plt.subplot(2, 1, 1)
    plt.imshow(image_np)  # PIL images are already in RGB order
    plt.axis('off')
    plt.title("Detected Text Regions", fontsize=12)
    
    # Add bounding boxes
    for bbox in bboxes:
        x, y, w, h = bbox
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))
    
    # Display predicted text in a separate subplot below
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.text(0.5, 0.5, f"Predicted Text: {pred_text}", fontsize=12, ha='center', va='center', wrap=True, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    plt.tight_layout()
    plt.show()


def visualize_attention_maps(model, crop_tensor, pred_text, attn_maps, figsize=(15, 8), tgt_seq=None):
    """
    Visualize attention maps for each character in the predicted text.
    
    Args:
        model: The transformer model with a CNN backbone
        crop_tensor: The input image tensor (already normalized)
        pred_text: The predicted text string
        attn_maps: List of attention maps from the transformer model
        figsize: Size of the figure to display
    """
    # Get the last layer's attention maps (typically most interpretable)
    last_attn = attn_maps[-1][0]  # Shape: [seq_len, src_len]
    
    # Convert tensor to numpy and denormalize for display
    with torch.no_grad():
        img_np = crop_tensor.cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        img_denorm = (img_np * std + mean) * 255
        img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
        img_display = np.transpose(img_denorm, (1, 2, 0))
    
    # Get feature map dimensions from the model's CNN
    with torch.no_grad():
        features = model.cnn(crop_tensor.unsqueeze(0).cuda())
        _, _, H, W = features.shape
    
    # Verify attention map source length matches expected dimensions
    attn_src_len = last_attn.shape[1]  # src_len: number of source tokens
    expected_len = H * W
    if attn_src_len != expected_len:
        print(f"Warning: Attention map source length ({attn_src_len}) does not match expected CNN output size ({expected_len}). Attempting to adjust...")
        # If the attention map size doesn't match, we interpolate or pad
        attn_weights_full = np.zeros((last_attn.shape[0], expected_len))
        for i in range(last_attn.shape[0]):
            attn_weights = last_attn[i].cpu().numpy()
            if attn_src_len < expected_len:
                # Pad with zeros
                attn_weights_full[i, :attn_src_len] = attn_weights
            else:
                # Truncate or interpolate (here we truncate for simplicity)
                attn_weights_full[i] = attn_weights[:expected_len]
    else:
        attn_weights_full = last_attn.cpu().numpy()
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display original image
    plt.subplot(1, len(pred_text) + 1, 1)
    plt.imshow(img_display)
    plt.title('Input Image')
    plt.axis('off')
    
    # Process each character's attention map
    for i, char in enumerate(pred_text):
        plt.subplot(1, len(pred_text) + 1, i + 2)
        
        # Get attention weights for this character
        # Add 1 to skip <SOS> token
        char_idx = i
        if char_idx < attn_weights_full.shape[0]:  # Ensure we don't go out of bounds
            attn_weights = attn_weights_full[char_idx]
            
            # Reshape to feature map dimensions
            heatmap = attn_weights.reshape(H, W)
            
            # Normalize heatmap for better visualization
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Resize to image dimensions for overlay
            heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
            
            # Convert to colormap
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = img_display.copy()
            alpha = 0.6  # Transparency factor
            cv2.addWeighted(heatmap_colored, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Display
            plt.imshow(overlay)
            plt.title(f"'{char}'")
            plt.axis('off')
        else:
            plt.imshow(img_display)
            plt.title(f"'{char}' (no attn)")
            plt.axis('off')
    
    plt.suptitle(f"Attention Maps for: '{pred_text}'", fontsize=16)
    plt.tight_layout()
    plt.show()
