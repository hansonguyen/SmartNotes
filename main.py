from ocr import detect_document
from enhancer import enhance

if __name__ == "__main__":
    IMAGE_NAME = input("Enter file name (must be in IMAGES directory): ")
    TOPICS = input("Enter list of related topics (separate by /): ")
    try:
        output = detect_document(IMAGE_NAME)
        enhanced_output = enhance(IMAGE_NAME, output, TOPICS)
        print(enhanced_output)

    except Exception as e:
        print(e)