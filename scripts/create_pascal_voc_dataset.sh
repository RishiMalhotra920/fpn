mkdir pascal_voc_data
cd pascal_voc_data
echo "Start time: $(date)"
cat > open_up.py << EOF
import torchvision
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

def create_datasets(root_dir: str) -> None:
    train_dataset = datasets.VOCDetection(root=root_dir,
                                          year='2012',
                                          image_set="train",
                                          transform=None,
                                          download=True)

    val_dataset = datasets.VOCDetection(root=root_dir,
                                        year='2012',
                                        image_set="val",
                                        transform=None,
                                        download=True)
    

create_datasets("/root/pascal_voc_data")
EOF
python open_up.py
echo "Time after opening up the dataset: $(date)"
echo "Dataset is ready"