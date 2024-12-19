# Point-JEPA: Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Clouds 🚀

This repository provides an implementation of **Point-JEPA**, a Joint Embedding Predictive Architecture tailored for self-supervised learning on point cloud data. The training pipeline is optimized for high-performance GPUs, such as the NVIDIA A100, to achieve state-of-the-art results.

---

## 🚦 Training Process

### Datasets: Prepare and configure the necessary datasets as described below.

---

## 📂 Datasets

Point-JEPA utilizes several datasets for training and evaluation. Below are the details and setup instructions for each:

### 1. **ModelNet40** 🏠

- **Description**: A benchmark dataset comprising 3D CAD models across 40 categories, such as chairs, tables, and airplanes.
- **Details**:
  - **Number of Classes**: 40
  - **Samples**: 12,311 models divided into training and testing sets.
- **Setup**:
  1. Download the dataset:
     ```bash
     cd data
     wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
     unzip modelnet40_ply_hdf5_2048.zip
     ```
  2. Ensure the data is organized as follows:
     ```
     data/
     └── modelnet40_ply_hdf5_2048/
         ├── ply_data_train0.h5
         ├── ply_data_train1.h5
         ├── ...
         └── shape_names.txt
     ```

### 2. **ShapeNet** 🛠️

- **Description**: A comprehensive dataset of 3D models categorized into various object classes.
- **Details**:
  - **Categories**: Multiple, including furniture, vehicles, and more.
- **Setup**:
  1. Download the dataset:
     - Obtain `ShapeNet55.zip` from the official source and place it in the `data` directory.
  2. Extract and organize:
     ```bash
     cd data
     unzip ShapeNet55.zip
     cp ../.metadata/ShapeNet55/* ShapeNet55/
     ```
  3. Process the dataset:
     ```bash
     python -m pointjepa.datasets.process.shapenet_npz
     ```
  4. The directory structure should be:
     ```
     data/
     └── ShapeNet55/
         ├── shapenet_train.npz
         ├── shapenet_test.npz
         └── ...
     ```

### 3. **ScanObjectNN** 🖼️

- **Description**: A dataset containing real-world 3D object scans, providing a diverse set of object point clouds.
- **Setup**:
  1. Download the dataset:
     - Agree to the Terms of Use on the official website and download `h5_files.zip`.
  2. Extract and organize:
     ```bash
     cd data
     unzip h5_files.zip
     mv h5_files ScanObjectNN
     ```
  3. Ensure the data is structured as:
     ```
     data/
     └── ScanObjectNN/
         ├── main_split/
         ├── main_split_nobg/
         └── ...
     ```

---

## ⚙️ Configuration

Adjust the training parameters in the configuration file (`configs/Point-JEPA/pretraining/shapenet.yaml`) as needed. Key parameters include:

- **Batch Size**: Optimal value depends on GPU memory; for A100, a batch size of 64 is recommended.
- **Learning Rate**: Default is `0.001` with cosine decay.
- **Epochs**: Set according to dataset size and desired performance; for example, 500 epochs for ModelNet40.

Example configuration snippet:

```yaml
batch_size: 64
learning_rate: 0.001
epochs: 500
dataset_path: /path/to/dataset
```
## 🔥 Training Steps

1. **Start Training**:
   ```bash
   python -m pointjepa fit -c configs/Point-JEPA/pretraining/shapenet.yaml
```

After running the above command:

Logs: Training progress and metrics will be saved in the `logs/` directory.  
Checkpoints: Model weights will be saved in the `artifacts/` directory after each epoch.

2. **Monitor Training**:  
Launch TensorBoard to visualize training metrics:  
```bash
tensorboard --logdir logs/
``` 
Additionally, monitor GPU utilization using:  
```bash
nvidia-smi
```

3. **Evaluation**:  
Once training is complete, evaluate the model's performance:  
```bash
python evaluate.py --model_path artifacts/best_model.pth
```  
This will output metrics such as accuracy and inference time.

---

## 🏆 Key Features

- **Self-Supervised Learning**: Employs joint embeddings for effective representation learning without the need for labeled data.
- **Efficiency**: Designed to reduce pre-training time compared to traditional methods.
- **Flexibility**: Supports various point cloud datasets with configurable parameters.

---

## 🤝 Acknowledgements

This implementation is based on the research paper [Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud](https://arxiv.org/abs/2404.16432) by **Ayumu Saito**, **Prachi Kudeshia**, and **Jiju Poovvancheri**. Special thanks to their original GitHub repository [Point-JEPA](https://github.com/Ayumu-J-S/Point-JEPA) for providing foundational resources.


---

## 📸 Training Process Screenshot

Below is a screenshot showcasing the training process conducted on Google Colab:

![Training Process in Colab](https://github.com/user-attachments/assets/f2260cfe-b9fe-4e1b-9004-84edb60aebef)
