import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# KONFIGURASI MODEL HUGGING FACE
# =====================================================

MODEL_NAME = "m467eeee/indobert-tweet_pilkada"

# Load tokenizer dan model sekali saat aplikasi dijalankan
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Set model ke mode evaluasi
model.eval()


def deteksi_pelanggaran(teks_bersih):
    """
    Melakukan prediksi teks menggunakan model IndoBERT yang telah dilatih.
    """

    # Tokenisasi
    inputs = tokenizer(
        teks_bersih,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Ambil logits
    logits = outputs.logits

    # Softmax -> Probabilitas
    probabilitas = torch.nn.functional.softmax(
        logits,
        dim=-1
    )

    # Ambil kelas dengan probabilitas tertinggi
    kelas_prediksi = torch.argmax(
        probabilitas,
        dim=1
    ).item()

    # Mapping label
    label_map = {
        0: "Bukan Pelanggaran",
        1: "Potensi Pelanggaran"
    }

    hasil_label = label_map.get(
        kelas_prediksi,
        "Tidak Diketahui"
    )

    confidence = (
        probabilitas[0][kelas_prediksi].item()
        * 100
    )

    return hasil_label, confidence