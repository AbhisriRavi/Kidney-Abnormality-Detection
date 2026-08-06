"""Patient-level label resolution rules."""
import numpy as np
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
# Every KiTS23 case is a tumour patient, but slices where the tumour is not in
# cross-section were labelled Normal. The modal label would therefore relabel
# tumour patients as Normal. Any abnormality outranks Normal.
LABEL_PRIORITY = ["Tumor", "Cyst", "Stone", "Normal"]
PRIORITY_IDX = [CLASSES.index(c) for c in LABEL_PRIORITY]
def resolve_patient_label(v, rule="priority"):
    """priority: most severe finding present. modal: most frequent label."""
    present = set(int(x) for x in v)
    if rule == "modal":
        u, c = np.unique(list(v), return_counts=True)
        return int(u[c.argmax()])
    for i in PRIORITY_IDX:
        if i in present:
            return i
    return int(list(present)[0])
