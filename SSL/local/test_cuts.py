import sys
from lhotse import CutSet

cuts = CutSet.from_file(sys.argv[1])

bad = []
checked = 0

for c in cuts:
    checked += 1
    try:
        feats = c.load_features()

        if feats is None or feats.shape[0] == 0:
            print(f"EMPTY: {c.id} shape={None if feats is None else feats.shape}")
            bad.append(c.id)
            continue

        # sanity: frames vs duration (10ms frame shift)
        expected = int(c.duration / 0.01)
        if abs(feats.shape[0] - expected) > 20:
            print(f"MISMATCH: {c.id} frames={feats.shape[0]} expected~{expected}")

    except Exception as e:
        print(f"BROKEN: {c.id} error={e}")
        bad.append(c.id)

    if checked % 1000 == 0:
        print(f"Checked {checked}")

print("\n==== SUMMARY ====")
print("Checked:", checked)
print("Bad:", len(bad))