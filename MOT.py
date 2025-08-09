from run_pipeline import main as run_pipeline

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--yolo_model', nargs='+', type=Path, default=[ROOT / 'yolo_model.pt'], 
                        help='YOLO model Path')
    parser.add_argument('--reid_model', nargs='+', type=Path, default=[ROOT / 'osnet_x0_25_msmt17.pt'],
                        help='ReID model Path')
    parser.add_argument('--input', type=str, default='example',
                        help='Input directory Name')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs',
                        help='Outputs saved in output/name')
    parser.add_argument('--name', type=str, default='example',
                        help='Outputs saved in output/name')
    
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
