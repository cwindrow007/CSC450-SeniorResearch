from pathlib import Path

input_dir = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_RAW")
output_dir = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_DECRYPTED")
output_dir.mkdir(parents=True, exist_ok=True)

KEY = 0xAA  # Sega’s simple XOR key

for dsc in input_dir.glob("*.dsc"):
    data = bytearray(dsc.read_bytes())
    for i in range(len(data)):
        data[i] ^= KEY  # XOR each byte

    out = output_dir / dsc.name
    out.write_bytes(data)
    print(f"Decrypted: {dsc.name} -> {out.name}")

print("\n Decryption complete. Files saved to:", output_dir)
