from pathlib import Path

# change this to the exact path of ONE .dsc file to inspect
f = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_RAW\pv_001_extreme_1.dsc")

data = f.read_bytes()
print(f"File size: {len(data)} bytes\n")

# show first 96 bytes after header start (0x40)
for i in range(0, 0x200, 16):
    chunk = " ".join(f"{b:02X}" for b in data[i:i+16])
    print(f"{i:04X}: {chunk}")
