from pathlib import Path
import argparse
from natsort import natsorted
from datetime import datetime

from ultralytics import YOLO
from Reid_models import REID_MODELS, WEIGHTS_URLS
import urllib.request, zipfile

import torchreid
import cv2
import torch

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "Inputs"
OUTPUTS = ROOT / "Outputs"
DETECTION_MODELS = ROOT / "Detection_models"
EMBEDDING_MODELS = ROOT / "Embedding_models"

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--input', type=str, default='meme',
                    help='Input directory name')
parser.add_argument('--yolo_model', nargs='+', type=str, default=['yolov8n.pt'],
                    help='YOLO model name')
parser.add_argument('--reid_model', nargs='+', type=str, default=['osnet_x0_25_market1501.pt'],
                    help='ReID model name')
parser.add_argument('--output', type=Path, default='run',
                        help='Output directory name')
args = parser.parse_args()

image_folder = ROOT / "Inputs" / Path(args.input) # dossier contenant les images # print(natsorted(os.listdir(image_folder)))
image_paths = natsorted(image_folder.iterdir())

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = OUTPUTS / Path(args.output) / Path(timestamp)

args.yolo_model = [DETECTION_MODELS / Path(m) for m in args.yolo_model]
args.reid_model = [EMBEDDING_MODELS / Path(m) for m in args.reid_model]

# Déplacement sur GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle YOLO via ultralytics
yolo = YOLO(args.yolo_model[0])

# Charger modèle ReID via torchreid
reid_model_weights_path = args.reid_model[0]
reid_model_weights_name = reid_model_weights_path.name

for name in REID_MODELS:
    if name in reid_model_weights_name:
        reid_model_name = name
        break

# print(f"Using ReID model: {reid_model_name}")
reid_model_class = getattr(torchreid.models, reid_model_name)
reid_model = reid_model_class(num_classes=1000)

if not reid_model_weights_path.exists():
    reid_model_weights_url = WEIGHTS_URLS.get(reid_model_weights_name)
    if reid_model_weights_url is None:
        print(f"No URL found for ReID model weights: {reid_model_weights_name}, please check the WEIGHTS_URLS dictionary. "
              f"Torchreid's default weight file {reid_model_name}_imagenet.pth has been used instead.")
        args.reid_model[0] = EMBEDDING_MODELS / Path(f"{reid_model_name}_imagenet.pth")
        print(f"Using default weights: {args.reid_model[0]}")
    else:
        urllib.request.urlretrieve(reid_model_weights_url, reid_model_weights_path)
        torchreid.utils.load_pretrained_weights(reid_model, reid_model_weights_path)
else:
    torchreid.utils.load_pretrained_weights(reid_model, reid_model_weights_path)
reid_model = reid_model.to(device)
reid_model.eval()

# Charger modèle de Tracker via deep_sort_realtime
tracker = DeepSort(
    max_age=30,       # frames sans match avant suppression
    n_init=3,         # confirmations requises
    nn_budget=100,    # taille de la galerie
    max_iou_distance=0.7
)

output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "detections").mkdir(parents=True, exist_ok=True)
(output_dir / "embeddings").mkdir(parents=True, exist_ok=True)
(output_dir / "tracking").mkdir(parents=True, exist_ok=True)
(output_dir / "boxes").mkdir(parents=True, exist_ok=True)

TRACKEVAL_INPUTS = ROOT / "TrackEval" / "data" / Path(args.input)
TRACKEVAL_INPUTS_TRACKERS = TRACKEVAL_INPUTS / "trackers" \
    / Path(args.yolo_model[0].stem + "_" + args.reid_model[0].stem + "_deepsort") \
    / "data" / "sequence_1"
TRACKEVAL_INPUTS_TRACKERS.mkdir(parents=True, exist_ok=True)

# Extraire les embeddings OSNet
def extract_embeddings(img, boxes):
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)                                  # Conversion des coordonnées bbox
        crop = img[y1:y2, x1:x2]                                        # Découpe de l'image
        crop = cv2.resize(crop, (128, 256))                             # Redimensionne (W, H)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)                    # OpenCV → RGB
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        crop = (crop - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
               torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crops.append(crop)

    if not crops:
        return []
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        embeddings = reid_model(batch)
    return embeddings.cpu().numpy()

tracking_file_global = TRACKEVAL_INPUTS_TRACKERS / "tracks.txt"
tracks_f = open(tracking_file_global, "w")

all_embeddings = []
all_boxes = []
frame_id = 1
for img_path in image_paths:

    img = cv2.imread(img_path) # Charger image
    if img is None:
      raise FileNotFoundError(
        f"Impossible de charger l'image : {img_path}. ")

    results = yolo(img, device=device, classes=[0])[0] # Exécuter YOLO

    boxes = results.boxes.xyxy.cpu().numpy() # Coordonnées des boîtes (x1, y1, x2, y2)
    confs = results.boxes.conf.cpu().numpy() # Confidence scores
    classes = results.boxes.cls.cpu().numpy()
    class_ids = classes.astype(int)
    class_names = yolo.model.names
    detected_labels = [class_names[c] for c in class_ids]

    embs = extract_embeddings(img, boxes)

    image_id = img_path.stem
    detection_file = output_dir / "detections" / f"{image_id}.txt"

    with detection_file.open("w") as f:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            w = x2 - x1
            h = y2 - y1
            score = confs[i]
            cls = class_ids[i]
            track_id = -1 # pas encore attribué
            f.write(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},{cls},1\n")

    # --- DeepSORT update (avec features fournis) ---
    # DeepSort attend des TLWH
    detections = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        tlwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        conf = float(confs[i])
        feat = embs[i].astype(np.float32) if len(embs) else None
        # Selon versions, DeepSort accepte:
        #  - tuple (tlwh, conf, feature)
        #  - ou dict {"bbox": tlwh, "confidence": conf, "feature": feat}
        detections.append((tlwh, conf, feat))

        tracks = tracker.update_tracks(detections, frame=img)  # img pour l'update interne (si nécessaire)

    tracking_file = output_dir / "tracking" / f"{image_id}.txt"
    with tracking_file.open("w") as f:
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = t.to_tlbr()
            w, h = x2 - x1, y2 - y1
            f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.0,0,1\n")
            tracks_f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.0,-1,-1\n")

    if len(embs):
        all_embeddings.append(embs)
        all_boxes.extend([[frame_id, *b] for b in boxes])

    frame_id += 1

tracks_f.close()
print(f"Fichier de tracking TrackEval écrit dans : {tracking_file_global}")

# Concaténer tous les embeddings extraits
if all_embeddings:
    all_embeddings_array = np.vstack(all_embeddings)
    np.save(output_dir /  "embeddings" / "embeddings.npy", all_embeddings_array)
    print(f"Embeddings sauvegardés dans : {output_dir / "embeddings" / 'embeddings.npy'}")

    # Sauvegarder les boîtes associées aux embeddings (optionnel mais utile)
    np.savetxt(output_dir / "boxes" / "boxes.txt", np.array(all_boxes), fmt="%.2f", delimiter=",")
    print(f"Coordonnées associées sauvegardées dans : {output_dir / "boxes" / 'boxes.txt'}")


# python TrackEval/scripts/run_mot_challenge.py \
#   --DATASET MOT17 \
#   --SPLIT_TO_EVAL train \
#   --TRACKERS_TO_EVAL MyTracker \
#   --SEQ_TO_EVAL MOT17-02-FRCNN


# # Downloading TrackEval
#     trackeval_url = "https://github.com/JonathonLuiten/TrackEval/archive/refs/heads/main.zip"
#     dest_path = ROOT / "TrackEval"
#     zip_path = dest_path.with_suffix(".zip")
#     extract_dir = dest_path.parent

#     if not dest_path.exists():
#         print(f"[INFO] Downloading TrackEval from GitHub...")
#         urllib.request.urlretrieve(trackeval_url, zip_path)
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_dir)
#             Path("TrackEval-master").rename(dest_path.name)
#             zip_path.unlink()







# print(results.boxes)

# annotated_img = results.plot()
# plt.imshow(annotated_img)
# plt.axis('off')
# plt.title("Détections YOLOv8")
# plt.show()

# # Coordonnées des boîtes (x1, y1, x2, y2)
# boxes = results.boxes.xyxy.cpu().numpy()
# print("Bounding boxes (xyxy):")
# print(boxes)

# # Scores de confiance
# confs = results.boxes.conf.cpu().numpy()
# print("\nConfidence scores:")
# print(confs)

# # Classes détectées (sous forme d'indices)
# classes = results.boxes.cls.cpu().numpy()
# class_ids = classes.astype(int)
# print("\nClass indices:")
# print(class_ids)

# # Noms des classes détectées
# class_names = yolo.model.names
# detected_labels = [class_names[c] for c in class_ids]
# print("\nClass names:")
# print(detected_labels)
