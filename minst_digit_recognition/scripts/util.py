import csv

from numpy import argmax


def render_submission(data):
    with open("data/submission.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId", "Label"])
        for i in range(len(data)):
            writer.writerow([i+1, argmax(data[i])])

