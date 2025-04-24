import argparse
from model import DeepfakeDetector

def main():
    parser = argparse.ArgumentParser(description='Ethical Deepfake Defence System')
    parser.add_argument('image_path', type=str, help='Path to the image file to classify')
    args = parser.parse_args()

    detector = DeepfakeDetector()
    label, confidence = detector.predict(args.image_path)
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")

if __name__ == '__main__':
    main()
