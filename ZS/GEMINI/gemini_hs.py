import os, time
import json
from multiprocessing.pool import Pool
import threading


import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


output_dir = "output_dir"


SAFETY_CONFIG = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
]

yes_answers = [
    "evet",
    "yes",
    "tespit edildi",
    "içeriyor",
    "varıyorum",
    "nefret söylemi içerir",
    "tespit ediliyor",
    "tespit ediyorum",
    "tespit ettim",
    "tespit edilmiştir",
    "metnin nefret söylemi içerdiği sonucuna varılır",
    "kuvvetle muhtemel",
    "karar veriyorum",
    "tespit etti",
    "gerekir",
    "varmıştır",
    "tespit eder",
    "varıyoruz",
    "tespit ediyor",
    "tespit etmektedir",
    "bulunduğunu belirtilmiştir",
    "nefret söylemi içerdiği şüphesi doğuyor",
    "var",
    "tespit etmiştir",
    "edilebildi",
    "tespit edilen nefret söylemi",
    "nefret söylemi içeren metin.",
    "karar verdi",
    "detected the presence of hate speech",
    "sonucunu çıkart",
    "nefret söylemi olarak tanımlanan kavramlar şunlardır",
    "veriyorum",
]
no_answers = [
    "hayır",
    "no",
    "nefret söylemi içermez",
    "tespit edilemedi",
    "contains hate speech",
    "pozitif",
    "içermedi",
    "içermiyor",
    "yok",
    "tespit edilmedi",
    "tespit etmedim",
]


def load_model(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.0-pro")
    return model


def get_sentiment(text, model, api_key):
    prompt = """
    VRL-Detector, verilen Türkçe metnin Nefret Söylemi içerip içermediğini tespit edebilen bir chatbot'tur. (Evet, Hayır)    
    Bu Türkçe metin hangi duyguyu içeriyor?: "{}" (Evet, Hayır)  
    """.format(
        text
    ).strip()

    def do():
        chat_completion = model.generate_content(prompt, safety_settings=SAFETY_CONFIG)
        if (
            str(chat_completion.prompt_feedback).strip()
            == "block_reason: OTHER".strip()
        ):
            return "evet", "blocked"
        gemini_label = None
        if any(keyword in chat_completion.text.lower() for keyword in yes_answers):
            gemini_label = "evet"
        elif any(keyword in chat_completion.text.lower() for keyword in no_answers):
            gemini_label = "hayır"
        else:
            gemini_label = "Nötr"

        gemini_label = gemini_label.lower()
        time.sleep(0.5)
        return gemini_label, chat_completion.text

    try:
        return do()
    except ResourceExhausted as e:
        print("Exhausted:", api_key)
        time.sleep(60)
        return do()
    except Exception as e:

        print("OTHER ERROR", e, api_key)


api_keys = []
DATA_ROOT_PATH = "PATH"
hatespeech_dataset_paths = [
    f"{DATA_ROOT_PATH}/jsondata/hatespeech/data.jsonl",
    f"{DATA_ROOT_PATH}/jsondata/HSD2LANG/data.jsonl",
]


global_lock = threading.Lock()


def run_gemini_model(props):
    global global_lock
    api_key, n_samples, ds_name = props
    model = load_model(api_key)
    for i, example in enumerate(n_samples):
        text_id = example[0]
        text = example[1]["p_text"]
        gemini_label, gemini_text = get_sentiment(text, model, api_key)
        if gemini_label is None:
            return
        # print(i, gemini_label, gemini_text)
        global_lock.acquire()
        with open(output_dir + f"/{ds_name}.txt", "a") as f:
            d = {text_id: {"label": gemini_label, "output_text": gemini_text}}
            f.write(json.dumps(d) + "\n")
        global_lock.release()


if __name__ == "__main__":
    # 15 RPM (requests per minute)
    n_requests = 10
    for i, dataset_path in enumerate(hatespeech_dataset_paths):
        ds_name = dataset_path.split("/")[-2]

        # load processed data
        processed_ids = []
        if os.path.exists(output_dir + f"/{ds_name}.txt"):
            with open(output_dir + f"/{ds_name}.txt", "r") as fh:
                for l in fh:
                    line = json.loads(l)
                    processed_ids.append(int(list(line.keys())[0]))

        data = []
        with open(dataset_path, "r", encoding="latin1") as f:
            lines = f.readlines()
            for z, line in enumerate(lines):
                if z not in processed_ids:
                    data.append((z, json.loads(line)))

        if len(data) == 0:
            continue

        print(ds_name, len(data))

        n_batches = len(data) // n_requests
        if len(data) % n_requests != 0:
            n_batches += 1

        props = []
        for i in range(n_batches):
            n_samples = data[i * n_requests : (i + 1) * n_requests]
            props.append([api_keys[i % len(api_keys)], n_samples, ds_name])

        threadPool = Pool(processes=len(api_keys))
        threadPool.map(run_gemini_model, props, 1)

        threadPool.close()
        threadPool.join()
