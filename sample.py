import os
import shutil
import random

def sample_images(src_root, dst_root, classes, samples_per_class):
    os.makedirs(dst_root, exist_ok=True)
    for cls in classes:
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_dir, exist_ok=True)

        all_images = os.listdir(src_dir)
        all_images = [f for f in all_images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(all_images) < samples_per_class:
            print(f"⚠️ Not enough images in {cls}, found {len(all_images)}")
            samples = all_images
        else:
            samples = random.sample(all_images, samples_per_class)

        for file in samples:
            shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        print(f"✅ Copied {len(samples)} images for class '{cls}'")

# Define classes and paths
classes = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
train_src = 'FairFace Race/train'
val_src = 'FairFace Race/val'
train_dst = 'fairface-7class-small/train'
val_dst = 'fairface-7class-small/val'

# Sample 500 for train, 100 for val
sample_images(train_src, train_dst, classes, 500)
sample_images(val_src, val_dst, classes, 100)
