

def classify_3classes(label, Threshold):
    if label > Threshold:
        return 2
    if label < -Threshold:
        return 1
    return 0


def default_classifier(label):
    return label