"""
main executable script
executes fir, score and confusion_matrix scripts in sequence
"""


def main():
    from mnist_digit_recognition import fit, score, confusion_matrix
    fit.main()
    score.main()
    confusion_matrix.main()


if __name__ == "__main__":
    main()
