import os
import argparse
import pydicom
import numpy as np
from PIL import Image

def convert_dicom_to_png(dicom_path, png_path, dpi=(300, 300)):
    ds = pydicom.dcmread(dicom_path)

    if "PixelData" not in ds:
        print(f"Arquivo {dicom_path} não contém dados de imagem.")
        return

    pixel_array = ds.pixel_array

    # Normalizar para 0–255 (8 bits)
    pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
    if pixel_max != pixel_min:
        pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0

    image = Image.fromarray(pixel_array.astype(np.uint8))
    image.save(png_path, "PNG", dpi=dpi)


def main():
    parser = argparse.ArgumentParser(description="Converter DICOM ECG para PNG sem anonimização.")
    parser.add_argument("--input", default="../data/ecg_images/dicom", help="Diretório de entrada com DICOMs")
    parser.add_argument("--output", default="../data/ecg_images/png", help="Diretório de saída para PNGs")
    parser.add_argument("--dpi", type=int, nargs=2, default=[300, 300],
                        help="Resolução em DPI da imagem exportada. Ex: --dpi 300 300")
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

                print(f"Convertendo {dicom_path} -> {png_path} com DPI={args.dpi}")
                convert_dicom_to_png(dicom_path, png_path, dpi=tuple(args.dpi))


if __name__ == "__main__":
    main()
