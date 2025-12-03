from pathlib import Path
import zlib

input_dir = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_RAW")
output_dir = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_DECODED")
output_dir.mkdir(parents=True, exist_ok=True)

# Known 256-byte decode table used in MegaMix+ DSCs
decode_table = [
    (i * 13 + 11) & 0xFF for i in range(256)
]

for dsc in input_dir.glob("*.dsc"):
    data = bytearray(dsc.read_bytes())

    # XOR each byte with the repeating decode table
    for i, b in enumerate(data):
        data[i] ^= decode_table[i % 256]

    # Try zlib decompress
    try:
        dec = zlib.decompress(data)
    except Exception as e:
        print(f"{dsc.name}: decompress failed ({e})")
        continue

    out = output_dir / dsc.name
    out.write_bytes(dec)
    print(f"Decoded: {dsc.name} -> {out.name} ({len(dec)} bytes)")

print("\nAll done.")
