import re

def bersihkan_teks(text):
    # 1. Lakukan Case Folding (karena di Colab fungsi ini terpisah)
    text = str(text).lower()
    
    # 2. Proses Text Cleaning (persis seperti kodinganmu)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # hapus URL
    text = re.sub(r'@\w+', '', text)  # hapus mention
    text = re.sub(r'#\w+', '', text)  # hapus hashtag
    text = re.sub(r'\d+', '', text)  # hapus angka
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # hapus karakter non-ASCII (emotikon/simbol)
    text = re.sub(r'\s+', ' ', text)  # hapus spasi berlebih
    
    return text.strip()