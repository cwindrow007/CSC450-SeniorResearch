import struct
import json
from pathlib import Path
from event_map import EVENT_MAP
from button_map import BUTTON_MAP

#== CONFIG ==
input_dir = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_RAW")
output_dir = Path(r"C:\ResearchProject\CSC450-SeniorResearch\DSC_JSON")
output_dir.mkdir(parents=True, exist_ok=True)

def parse_dsc(path):
    """Parse Project DIVA MegaMix+ .dsc chart (12-byte event records)."""
    data = path.read_bytes()
    events = []

    # default skip after header
    header_skip = 0x80
    if data[:3] == b"DSC":
        header_skip = 0x80
    elif data[0:2] == b"\x00\x01":
        header_skip = 0x80

    # --- TRY small offsets (helps find true start) ---
    for start_offset in range(header_skip, header_skip + 12):
        event_id = int.from_bytes(data[start_offset:start_offset+2], "big")
        if event_id in (0x23, 0x24, 0x25, 0x26, 0x27):  # known valid codes
            header_skip = start_offset
            print(f"Found event section at 0x{header_skip:X}")
            break

    i = header_skip
    n = len(data)

    while i + 12 <= n:
        event_id, flags, time_raw, p1, p2 = struct.unpack_from(">HHIBB", data, i)
        if event_id == 0xFFFF:
            break

        evt_name = EVENT_MAP.get(event_id, f"UNK_{event_id:04X}")
        button = BUTTON_MAP.get(p1, f"UNK_{p1:02X}")

        events.append({
            "offset": i,
            "time": round(time_raw / 30000.0, 3),
            "event_id": event_id,
            "flags": flags,
            "type": evt_name,
            "button": button,
            "param1": p1,
            "param2": p2
        })
        i += 12

    return events


def convert_all():
    """CONVERSION OF ALL DSC FILES IN INPUT DIRECTORY"""

    files = sorted(input_dir.glob("*.dsc"))
    if not files:
        print("No DSC files found in input directory")
        return
    print(f"Found {len(files)} DSC files")

    for f in files:
        events = parse_dsc(f)
        out_path = output_dir / (f.stem + ".json")
        json.dump({"song": f.stem, "events": events}, open(out_path, "w"), indent=2)
        print(f"{f.name} -> {out_path.name} ({len(events)} events)")

    print ("\n CONVERSION DONE, JSON SAVED TO: ", output_dir)

if __name__ == "__main__":
     convert_all()