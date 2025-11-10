import torch
from pathlib import Path

def load_model_state(client_id):
    ckpt_path = Path(f"./outputs/client_{client_id:02d}/best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # ckpt 구조가 {"model": state_dict} 이거나 곧바로 state_dict 인 경우 둘 다 대응
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    print(f"[load] client_{client_id:02d} loaded with {len(state.keys())} params")
    return state

def average_states(state_dicts):
    # 모든 client 가 공통으로 가지고 있는 key만 사용
    common_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common_keys &= set(sd.keys())

    print(f"[average] common params = {len(common_keys)}")

    avg_state = {}
    for k in common_keys:
        tensors = [sd[k] for sd in state_dicts]
        avg_state[k] = sum(tensors) / len(tensors)
    return avg_state

def save_global_for_group(client_ids, out_path):
    if not client_ids:
        print(f"[skip] no clients for {out_path}")
        return
    print(f"\n=== FedAvg baseline for clients {client_ids} ===")
    state_dicts = [load_model_state(cid) for cid in client_ids]
    avg_state = average_states(state_dicts)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg_state, out_path)
    print(f"[FedAvg Baseline] Saved global averaged model → {out_path}")

def main():
    # 네 셋업 기준:
    # 1–16 : multimodal, 17–18 : image-only, 19–20 : text-only
    multimodal_ids = list(range(1, 17))
    image_only_ids = [17, 18]
    text_only_ids  = [19, 20]

    save_global_for_group(multimodal_ids, "./outputs/global_fedavg_multimodal.pt")
    save_global_for_group(image_only_ids, "./outputs/global_fedavg_imgonly.pt")
    save_global_for_group(text_only_ids,  "./outputs/global_fedavg_txtonly.pt")

if __name__ == "__main__":
    main()
