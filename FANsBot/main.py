from models import predict_author, device
from models.pytorch import author_bilstm

def main():
    while True:
        value = input("Enter the author text or (q) to quit: ")
        if value.strip().lower() == "q":
            break
        preds = predict_author(author_bilstm, value, device)
        top = preds["top"]

        print("="*50)
        print("📑 AUTHOR TEXT")
        print("="*50)
        print(value)
        print()
        print("="*50)
        print("🔮 TOP PREDICTION")
        print("="*50)
        print(f" Author       : {top['author']}")
        print(f" Class Code   : {top['class']}")
        print(f" Confidence   : {top['probability']*100:.2f}%")
        print()
        print("="*50)
        print("📊 ALL PREDICTIONS")
        print("="*50)
        for pred in preds["predictions"]:
            print(f"  - {pred['author']:15} ({pred['class']}) → {pred['probability']*100:.2f}%")
        print("="*50)
        print()

if __name__ == "__main__":
    main()
