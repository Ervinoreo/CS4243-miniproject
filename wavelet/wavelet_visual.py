import os
import cv2
import pywt
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")   # use non-GUI backend for speed
import matplotlib.pyplot as plt

plt.ioff()  # disable interactive mode for faster saving

# list of wavelets to apply
wavelets = ["haar", "db2", "sym2"]
max_level = 3 # number of decomposition levels

def extract_wavelet_features(arr):
    arr_abs = np.abs(arr)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "energy": np.sum(np.square(arr)),
        "entropy": -np.sum(np.where(arr_abs > 0, arr_abs * np.log2(arr_abs + 1e-12), 0)),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def wavelet_features(img, wavelet, levels=max_level):
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    features = {}
    subband_images = []
    for lvl, coeff in enumerate(coeffs):
        if lvl == 0:
            LL = coeff
            stats = extract_wavelet_features(LL)
            for k, v in stats.items():
                features[f"{wavelet}_L{lvl}_LL_{k}"] = v
            subband_images.append(LL)
        else:
            LH, HL, HH = coeff
            subbands = {"LH": LH, "HL": HL, "HH": HH}
            for name, band in subbands.items():
                stats = extract_wavelet_features(band)
                for k, v in stats.items():
                    features[f"{wavelet}_L{lvl}_{name}_{k}"] = v

            # combine subbands for visualization
            top = np.hstack((LH, HL))
            bottom = np.hstack((HH, np.zeros_like(HH)))
            combined = np.vstack((top, bottom))
            subband_images.append(combined)

    return features, subband_images


# function to get wavelet-transformed image
def wavelet_image(img, wavelet):
    coeffs2 = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs2
    # merge subbands into one visualization image
    top = np.hstack((LL, LH))
    bottom = np.hstack((HL, HH))
    combined = np.vstack((top, bottom))
    combined = cv2.resize(combined, (img.shape[1], img.shape[0]))
    return np.uint8(255 * (combined - combined.min()) / (np.ptp(combined) + 1e-8))

if __name__ == "__main__":
    # read input and output directories from the user
    parser = argparse.ArgumentParser(description="Apply wavelet transforms to images in a folder.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save transformed images.")
    args = parser.parse_args()
    input_root = args.input_folder
    output_root = args.output_folder

    labeled_root = os.path.join(output_root, "labeled")
    non_labeled_root = os.path.join(output_root, "non-labeled")
    feature_file = os.path.join(output_root, "wavelet_features.csv")
    os.makedirs(labeled_root, exist_ok=True)
    os.makedirs(non_labeled_root, exist_ok=True)

    img_count = 0
    all_features = []

    for root, dirs, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        labeled_dir = os.path.join(labeled_root, rel_path)
        non_labeled_dir = os.path.join(non_labeled_root, rel_path)
        os.makedirs(labeled_dir, exist_ok=True)
        os.makedirs(non_labeled_dir, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            in_path = os.path.join(root, fname)
            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_features = {"filename": fname}
            transformed_all = []
            
            # collect individual subband images for detailed visualization
            all_subband_images = []
            all_subband_titles = []

            # process each wavelet
            for w in wavelets:
                feats, subband_imgs = wavelet_features(img, w, levels=max_level)
                img_features.update(feats)

                # collect individual subbands for this wavelet
                wavelet_subbands = []
                wavelet_titles = []
                
                # Get individual subbands from the coefficients
                coeffs = pywt.wavedec2(img, w, level=max_level)
                for lvl, coeff in enumerate(coeffs):
                    if lvl == 0:
                        # LL subband (approximation)
                        LL = coeff
                        LL_norm = np.uint8(255 * (LL - LL.min()) / (np.ptp(LL) + 1e-8))
                        wavelet_subbands.append(LL_norm)
                        wavelet_titles.append(f"{w} L{lvl} LL")
                    else:
                        # Detail subbands
                        LH, HL, HH = coeff
                        for name, band in [("LH", LH), ("HL", HL), ("HH", HH)]:
                            band_norm = np.uint8(255 * (band - band.min()) / (np.ptp(band) + 1e-8))
                            wavelet_subbands.append(band_norm)
                            wavelet_titles.append(f"{w} L{lvl} {name}")
                
                all_subband_images.append(wavelet_subbands)
                all_subband_titles.append(wavelet_titles)

                # stack all level images vertically for visualization (for non-labeled output)
                vis_bands = []
                for sb in subband_imgs:
                    sb_norm = np.uint8(255 * (sb - sb.min()) / (np.ptp(sb) + 1e-8))
                    sb_resized = cv2.resize(sb_norm, (img.shape[1], img.shape[0]))  # match original height
                    vis_bands.append(sb_resized)
                combined_wavelet = np.vstack(vis_bands)

                # **Resize combined_wavelet to original height**
                combined_wavelet_resized = cv2.resize(combined_wavelet, (img.shape[1], img.shape[0]))
                transformed_all.append(combined_wavelet_resized)
            
            all_features.append(img_features)

            combined = np.hstack([img] + transformed_all)
            out_path = os.path.join(non_labeled_dir, fname)
            cv2.imwrite(out_path, combined)

            # labeled visualization for first few images
            if img_count < 5:
                # Calculate grid dimensions
                max_cols = 5
                
                # Prepare all images to display
                display_images = []
                display_titles = []
                
                # First row: original image (centered)
                display_images.append([img])
                display_titles.append(["Original"])
                
                # Add wavelet subbands row by row
                for wavelet_idx, (subbands, titles) in enumerate(zip(all_subband_images, all_subband_titles)):
                    # Split subbands into rows of max 8 columns
                    for i in range(0, len(subbands), max_cols):
                        row_images = subbands[i:i+max_cols]
                        row_titles = titles[i:i+max_cols]
                        display_images.append(row_images)
                        display_titles.append(row_titles)
                
                # Calculate total rows and max columns needed
                total_rows = len(display_images)
                max_cols_actual = max(len(row) for row in display_images)
                
                fig, axes = plt.subplots(total_rows, max_cols_actual, figsize=(2*max_cols_actual, 2*total_rows))
                fig.suptitle(f"Wavelet Multi-level Results - {fname}", fontsize=14, fontweight="bold")
                
                # Handle single row case
                if total_rows == 1:
                    axes = [axes]
                if max_cols_actual == 1:
                    axes = [[ax] for ax in axes]
                
                # Display images
                for row_idx, (row_images, row_titles) in enumerate(zip(display_images, display_titles)):
                    for col_idx in range(max_cols_actual):
                        if col_idx < len(row_images):
                            axes[row_idx][col_idx].imshow(row_images[col_idx], cmap="gray")
                            axes[row_idx][col_idx].set_title(row_titles[col_idx], fontsize=8)
                        axes[row_idx][col_idx].axis("off")
                
                plt.tight_layout()
                labeled_out_path = os.path.join(labeled_dir, f"labeled_{fname}")
                plt.savefig(labeled_out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            img_count += 1

    import pandas as pd
    df = pd.DataFrame(all_features)
    df.to_csv(feature_file, index=False)

    print("✅ Done! Output saved in:")
    print("   📁 Labeled:", labeled_root)
    print("   📁 Non-labeled:", non_labeled_root)
    print("   📄 Features CSV:", feature_file)
