import urllib.request, zipfile
from pathlib import Path
import copy
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
UL_MODELS = ["yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "rtdetr", "sam"]

def is_ultralytics_model(yolo_name):
    return any(yolo in str(yolo_name) for yolo in UL_MODELS)

def main(args):

    # Downloading TrackEval
    trackeval_url = "https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj"
    dest_path = ROOT / "TrackEval"
    zip_path = dest_path.with_suffix(".zip")
    extract_dir = dest_path.parent

    if not dest_path.exists():
        urllib.request.urlretrieve(trackeval_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            Path("TrackEval-master").rename(dest_path.name)
            zip_path.unlink()

    if not (ROOT / Path(args.input)).exists():
        print(f"Input directory {args.input} does not exist. Please provide a valid input directory.")
        return

    folders = sorted((ROOT / Path(args.input)).iterdir())
    

    dets = args.output / args.name / 'dets_n_embs' / args.yolo_model[0].stem / 'dets'
    embs = args.output / args.name / 'dets_n_embs' / args.yolo_model[0].stem / 'embs' / args.reid_model[0].stem
    
    if not (args.output / args.name).exists():
        dets.mkdir(parents=True, exist_ok=True)
        embs.mkdir(parents=True, exist_ok=True)

    new_opt = copy.deepcopy(args.opt)

    generate_dets_embs(new_opt, args.yolo_model[0], source= ROOT / Path(args.input) / 'img1')



    for i, folder in enumerate(folders):



        # tasks = [(y, p) for p in mot_folders if not (dets / f"{p.name}.txt").exists() or not (embs / f"{p.name}.txt").exists()]
        # with concurrent.futures.ProcessPoolExecutor(NUM_THREADS) as ex:
        #     futures = [ex.submit(process_single_det_emb, y, p, opt) for y, p in tasks]
        #     [LOGGER.error(f"Error: {e}") for f in concurrent.futures.as_completed(futures) if (e := f.exception())]
    
def generate_dets_embs(args: argparse.Namespace, y: Path, source: Path):
    
    yolo = YOLO(
        y if is_ultralytics_model(y)
        else 'yolov8n.pt',
    )

    results = yolo(
        source=source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        stream=True,
        device=args.device,
        verbose=False,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
    )
