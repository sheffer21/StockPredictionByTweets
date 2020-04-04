

def classify_3classes(label, Threshold):
    if label > Threshold:
        return 2
    if label < -Threshold:
        return 1
    return 0


def classify_Nclasses(label, n, min, max):
    # 2 Bins for lowers then min and greater then max and the other bins divided equably between max and min
    bin_size = (max-min) / (n-2)
    if label >= max:
        return n-1
    if label <= min:
        return 0

    for index in range(n-2):
        bin_lower = min + index * bin_size
        bin_upper = min + (index + 1) * bin_size
        if bin_lower < label < bin_upper:
            return index + 1

    raise Exception()

def default_classifier(label):
    return label
