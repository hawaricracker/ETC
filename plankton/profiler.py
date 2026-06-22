import os
import cv2
import time
import numpy as np
import onnx
import onnxruntime as ort
from collections import defaultdict
from onnx_tool import model_profile

idx_to_class = {
0:"Thalassionema",
1:"pennates_on_diatoms",
2:"kiteflagellates",
3:"mix_elongated",
4:"Thalassiosira",
5:"Cylindrotheca",
6:"Prorocentrum",
7:"Cerataulina",
8:"Strombidium_wulffi",
9:"pennate",
10:"Leptocylindrus",
11:"Chaetoceros",
12:"dino30",
13:"Chaetoceros_didymus_flagellate",
14:"Guinardia_delicatula",
15:"other_interaction",
16:"Katodinium_or_Torodinium",
17:"Gonyaulax",
18:"Rhizosolenia",
19:"Strombidium_morphotype1",
20:"Chrysochromulina",
21:"Skeletonema",
22:"Gyrodinium",
23:"Dictyocha",
24:"Ciliate_mix",
25:"G_delicatula_external_parasite",
26:"Pseudonitzschia",
27:"Dactyliosolen",
28:"diatom_flagellate",
29:"flagellate_sp3",
30:"pennate_morphotype1",
31:"Guinardia_striata",
32:"Heterocapsa_triquetra",
33:"bead",
34:"Strombidium_morphotype2",
35:"Corethron",
36:"DactFragCerataul",
37:"Proterythropsis_sp",
38:"Eucampia",
39:"Laboea_strobila",
40:"Coscinodiscus",
41:"G_delicatula_parasite",
42:"Ditylum",
43:"Thalassiosira_dirty",
44:"Tintinnid",
45:"Ceratium",
46:"Asterionellopsis",
47:"bad",
48:"Phaeocystis",
49:"Pyramimonas_longicauda",
50:"Strobilidium_morphotype1",
51:"Chaetoceros_didymus",
52:"Amphidinium_sp",
53:"Mesodinium_sp",
54:"Ephemera",
55:"Euglena",
56:"Chaetoceros_pennate",
57:"Guinardia_flaccida",
58:"Leptocylindrus_mediterraneus",
59:"Pleurosigma",
60:"Leegaardiella_ovalis",
61:"G_delicatula_detritus",
62:"Paralia",
63:"detritus",
64:"spore",
65:"Delphineis",
66:"Emiliania_huxleyi",
67:"Dinophysis",
68:"Tontonia_gracillima",
69:"amoeba",
70:"Lauderia",
71:"Strombidium_inclinatum",
72:"Chaetoceros_other",
73:"Ditylum_parasite",
74:"Dinobryon",
75:"Pseudochattonella_farcimen",
76:"clusterflagellate",
77:"Strombidium_oculatum",
78:"dino_large1",
79:"Licmophora",
80:"Chaetoceros_flagellate"
}

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img

model = "Yolo_final.onnx"

so = ort.SessionOptions()
so.enable_profiling = True

session = ort.InferenceSession(
    model,
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

dummy_rgb = np.random.randint(
    0,
    256,
    (200, 300, 3),
    dtype=np.uint8
)

dummy_input = preprocess(dummy_rgb)

session.run(None, {input_name: dummy_input})

profile_file = session.end_profiling()

print("Profiling file:", profile_file)

session1 = ort.InferenceSession(
    model,
    providers=["CPUExecutionProvider"]
)

input_name = session1.get_inputs()[0].name

for _ in range(50):
    h = np.random.randint(64, 512)
    w = np.random.randint(64, 512)

    img = np.random.randint(
        0,
        256,
        (h, w, 3),
        dtype=np.uint8
    )

    inp = preprocess(img)

    session1.run(None, {input_name: inp})

iterations = 10000

start = time.time()

for _ in range(iterations):
    h = np.random.randint(64, 512)
    w = np.random.randint(64, 512)

    img = np.random.randint(
        0,
        256,
        (h, w, 3),
        dtype=np.uint8
    )

    inp = preprocess(img)

    session1.run(None, {input_name: inp})

end = time.time()

fps = iterations / (end - start)

print("FPS (preprocess + inference):", fps)

model_onnx = onnx.load(model)

model_profile(
    model_onnx,
    {
        "input": np.random.randn(1, 1, 64, 64).astype(np.float32)
    }
)

folder = "gambar"

valid_ext = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

total = 0
correct = 0

class_total = defaultdict(int)
class_correct = defaultdict(int)

misclassified = []

for file in sorted(os.listdir(folder)):

    ext = os.path.splitext(file)[1].lower()

    if ext not in valid_ext:
        continue

    path = os.path.join(folder, file)

    img = cv2.imread(path)

    if img is None:
        continue

    inp = preprocess(img)

    output = session1.run(
        None,
        {input_name: inp}
    )[0]

    pred_idx = int(np.argmax(output))
    pred_class = idx_to_class[pred_idx]

    filename_no_ext = os.path.splitext(file)[0]

    parts = filename_no_ext.split("_", 1)

    if len(parts) != 2:
        continue

    true_class = parts[1]

    total += 1
    class_total[true_class] += 1

    if pred_class == true_class:
        correct += 1
        class_correct[true_class] += 1
    else:
        misclassified.append(
            (file, true_class, pred_class)
        )

print("\nHASIL PER KELAS")
print("-" * 90)

for cls in sorted(class_total.keys()):
    benar = class_correct[cls]
    total_kelas = class_total[cls]
    salah = total_kelas - benar
    akurasi = 100.0 * benar / total_kelas

    print(
        f"{cls:<35} "
        f"Benar={benar:<3d} "
        f"Salah={salah:<3d} "
        f"Akurasi={akurasi:6.2f}%"
    )

print("\nAKURASI KESELURUHAN")
print("-" * 90)

if total > 0:
    print(
        f"Benar={correct} "
        f"Salah={total-correct} "
        f"Akurasi={100.0*correct/total:.2f}% "
        f"({correct}/{total})"
    )

print("\nSALAH KLASIFIKASI")
print("-" * 90)

for file, true_class, pred_class in misclassified:
    print(
        f"{file} | "
        f"GT={true_class} | "
        f"PRED={pred_class}"
    )
