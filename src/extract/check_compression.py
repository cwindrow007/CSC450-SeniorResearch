from pathlib import Path
f = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_DECRYPTED\pv_001_extreme_1.dsc")
print(" ".join(f"{b:02X}" for b in f.read_bytes()[:16]))