import argparse, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help=r"예: C:\Users\user\PycharmProjects\ROCO\archive\rocov2\_prepared")
    ap.add_argument("--pattern", type=str, default="client_*/*.pt",
                    help=r"클라이언트 체크포인트 패턴 (기본: client_*/*.pt)")
    ap.add_argument("--dry", action="store_true", help="변환하지 않고 목록만 출력")
    args = ap.parse_args()

    root = Path(args.root)
    files = list(root.rglob(args.pattern))
    print(f"[info] root={root}")
    print(f"[info] found {len(files)} .pt files with pattern '{args.pattern}'")
    for i, p in enumerate(files[:10], 1):
        print(f"  {i:02d}) {p}")

    if args.dry or not files:
        return

    try:
        import torch
    except Exception as e:
        print("[error] torch import 실패:", e); sys.exit(1)
    try:
        from safetensors.torch import save_file as safe_save
    except Exception as e:
        print("[error] safetensors 미설치: pip install safetensors"); sys.exit(1)

    for pt in files:
        out = pt.with_suffix(".safetensors")
        if out.exists():
            print(f"[skip] already exists: {out}")
            continue
        try:
            # Torch 2.6+면 weights_only 사용 권장 (취약점 회피)
            sd = torch.load(str(pt), map_location="cpu", weights_only=True)
            if not isinstance(sd, dict) or not any(hasattr(v, "dtype") for v in sd.values()):
                print(f"[warn] not a plain state_dict: {pt} (스킵 또는 Torch>=2.6 필요)")
                continue
            safe_save(sd, str(out))
            print(f"[ok] {pt.name}  ->  {out.name}")
        except TypeError:
            print(f"[fail] Torch <2.6 환경: weights_only 지원 안 함 → 먼저 PyTorch 2.6+ 업그레이드 필요 ({pt})")
        except Exception as e:
            print(f"[fail] {pt}: {e}")

if __name__ == "__main__":
    main()
