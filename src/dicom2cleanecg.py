import os
import argparse
import numpy as np
import pydicom
import cv2
from PIL import Image, ImageDraw
import pytesseract

def load_dicom_as_image(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    if "PixelData" not in ds:
        raise ValueError(f"{dicom_path} não contém dados de imagem")

    pixel_array = ds.pixel_array.astype(np.float32)

    pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
    if pixel_max != pixel_min:
        pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0

    return ds, pixel_array.astype(np.uint8)

def deskew_image(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if len(angles) == 0:
        return image

    median_angle = np.median(angles)
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def normalize_contrast(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def anonymize_image(image, ds):
    """Apaga o nome do paciente na imagem com base no OCR e substitui por 'ANONYMOUS'."""
    if not hasattr(ds, "PatientName"):
        print("Aviso: DICOM sem campo PatientName.")
        return image

    patient_name_str = str(ds.PatientName).upper()
    name_parts = patient_name_str.split()
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    found_boxes = []

    for i, text in enumerate(ocr_data["text"]):
        if text.upper() in name_parts:
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            found_boxes.append((x, y, x + w, y + h))

    if found_boxes:
        min_x = min(box[0] for box in found_boxes)
        min_y = min(box[1] for box in found_boxes)
        max_x = max(box[2] for box in found_boxes)
        max_y = max(box[3] for box in found_boxes)
        redaction_box = [(min_x - 5, min_y - 2), (max_x + 5, max_y + 2)]

        draw.rectangle(redaction_box, fill="white")
    else:
        print(f"Aviso: não foi possível localizar '{patient_name_str}' via OCR.")

    return np.array(pil_image)

def convert_dicom_to_png(dicom_path, png_path, dpi=(300, 300), deskew=False, grayscale=True, anonymize=False):
    try:
        ds, image = load_dicom_as_image(dicom_path)

        if grayscale:
            image = normalize_contrast(image)
        if deskew:
            image = deskew_image(image)
        image = anonymize_image(image, ds)

        Image.fromarray(image).save(png_path, "PNG", dpi=dpi)
        print(f"Convertido: {dicom_path} -> {png_path}")
    except Exception as e:
        print(f"Erro em {dicom_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Converter DICOM ECG para PNG com pré-processamento e anonimização.")
    parser.add_argument("--input", default="../data/ecg_images/dicom", help="Diretório de entrada com DICOMs")
    parser.add_argument("--output", default="../data/ecg_images/png", help="Diretório de saída para PNGs")
    parser.add_argument("--dpi", type=int, nargs=2, default=[300, 300], help="Resolução em DPI da imagem exportada")
    parser.add_argument("--deskew", action="store_true", help="Aplicar correção de inclinação (Hough)")
    parser.add_argument("--grayscale", action="store_true", help="Converter para grayscale + normalizar contraste")
    parser.add_argument("--anonymize", action="store_true", help="Remover nome do paciente via OCR")
    args = parser.parse_args()

    for root, _, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, args.input)
                output_patient_dir = os.path.join(args.output, relative_path)
                os.makedirs(output_patient_dir, exist_ok=True)
                base_name = os.path.splitext(file)[0]
                png_path = os.path.join(output_patient_dir, f"{base_name}.png")

                convert_dicom_to_png(
                    dicom_path,
                    png_path,
                    dpi=tuple(args.dpi),
                    deskew=args.deskew,
                    grayscale=args.grayscale,
                    anonymize=args.anonymize,
                )

if __name__ == "__main__":
    main()
