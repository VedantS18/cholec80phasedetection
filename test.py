from dataset.cholec80_dataset import Cholec80ClipDataset

dataset = Cholec80ClipDataset(root="data/frames")

print("Total clips:", len(dataset))
if len(dataset) == 0:
    print("No clips found. Checking sample videos and labels...")
    import os
    video_dirs = sorted(os.listdir("data/frames"))
    print("Found videos:", video_dirs[:5])

    for video in video_dirs[:5]:
        path = f"data/frames/{video}"
        print(f"\n-- {video} --")
        frame_files = os.listdir(path)
        print(f"{len(frame_files)} frame(s)")
        example = frame_files[:3]
        print("Example frames:", example)

        label_path = f"data/phase_annotations/{video}-phase.txt"
        exists = os.path.exists(label_path)
        print(f"Label exists: {exists}")
