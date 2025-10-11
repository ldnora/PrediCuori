import os
import re
import base64
import shutil
import argparse
from cryptography.fernet import Fernet

def clean_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())

def encrypt_name(name: str, fernet: Fernet) -> str:
    encrypted = fernet.encrypt(name.encode())
    return base64.urlsafe_b64encode(encrypted).decode('utf-8').rstrip('=')

def decrypt_name(encrypted_name: str, fernet: Fernet) -> str:
    padded = encrypted_name + '=' * (-len(encrypted_name) % 4)
    decrypted = fernet.decrypt(base64.urlsafe_b64decode(padded.encode()))
    return decrypted.decode('utf-8')

def pseudoanonymize_folders(input_dir, output_dir, key_path):
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(key)
        print(f"[INFO] Nova chave gerada em: {key_path}")
    else:
        key = open(key_path, "rb").read()
    
    fernet = Fernet(key)
    os.makedirs(output_dir, exist_ok=True)

    for folder in os.listdir(input_dir):
        full_path = os.path.join(input_dir, folder)
        if not os.path.isdir(full_path):
            continue

        cleaned = clean_name(folder)
        anon_name = encrypt_name(cleaned, fernet)
        anon_path = os.path.join(output_dir, anon_name)

        shutil.copytree(full_path, anon_path, dirs_exist_ok=True)
        print(f"[OK] {folder} → {anon_name}")

def desanonimize_folders(input_dir, output_dir, key_path):
    if not os.path.exists(key_path):
        raise FileNotFoundError("Chave privada não encontrada!")

    key = open(key_path, "rb").read()
    fernet = Fernet(key)
    os.makedirs(output_dir, exist_ok=True)

    for folder in os.listdir(input_dir):
        full_path = os.path.join(input_dir, folder)
        if not os.path.isdir(full_path):
            continue

        try:
            original_name = decrypt_name(folder, fernet)
        except Exception:
            print(f"[WARN] Pasta {folder} não pôde ser decriptada.")
            continue

        restored_path = os.path.join(output_dir, original_name)
        shutil.copytree(full_path, restored_path, dirs_exist_ok=True)
        print(f"[OK] {folder} → {original_name}")

    print(f"\n[INFO] Desanonimização concluída!")

def main():
    parser = argparse.ArgumentParser(description="Pseudoanonimiza ou desanonimiza pastas de pacientes usando criptografia simétrica.")
    parser.add_argument("--input", default="./data/ecg_images/png", help="Diretório de entrada")
    parser.add_argument("--output", default="./data/ecg_images/png-anonymazed", help="Diretório de saída")
    parser.add_argument("--key", default="./src/private_key.key", help="Arquivo da chave privada")
    parser.add_argument("--reverse", action="store_true", help="Modo reverso: desanonimiza pastas")
    args = parser.parse_args()

    if args.reverse:
        desanonimize_folders(args.input, args.output, args.key)
    else:
        pseudoanonymize_folders(args.input, args.output, args.key)

if __name__ == "__main__":
    main()
