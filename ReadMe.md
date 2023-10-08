# Movie recommender system using AutoRec and SVD
## 📚 Table of Contents
- [Required Libraries](#required-libraries)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Data](#data)
- [Results](#results)
- [References](#references)
- [Training Parameters](#training-parameters)

## 📦 Required Libraries
- pandas
- numpy
- scikit-learn
- torch
- matplotlib

## 💻 Usage
### General Instructions
- **`main.py`**: 
  - Run using: `python main.py`.
  - Manages the implementation of input and output interfaces.
  - Note: `user_id` should range from 1 to 6040.

- **`Evaluation.py`**: 
  - Run using: `python Evaluation.py`.
  - Evaluates both AutoRec and Non-Personalised models.

### Model Testing (Requires code modification)
- **`CF_personalised.py`**: 
  - Uncomment line 144 and comment line 146 to test.
  - Run using: `python CF_personalised.py`.
  - Test the AutoRec model utilizing the trained model within the "model" folder.

- **`non_personalised.py`**: 
  - Uncomment line 94 and comment line 96 to test.
  - Run using: `python non_personalised.py`.
  - Test the Non-Personalised model.

### Training (Optional)
- **`AutoRecTraining.ipynb`**: 
  - Only necessary if training the AutoRec model from scratch.
  - Note: Pre-trained model available in the "model" folder.

## 📁 File Descriptions
- `main.py`: Entry point, manages input/output interfaces.
- `Evaluation.py`: Evaluates both AutoRec and Non-Personalised models.
- `AutoRecTraining.ipynb`: Used for training the AutoRec model (optional).
- `CF_personalised.py`: Implements and evaluates the AutoRec model.
- `non_personalised.py`: Implements and evaluates the Non-Personalised model.
- `bcolours.py`: Enables colored text output in terminal.
- `Evaluation_result.txt`: Contains results from model evaluations.
- `model`: Folder storing the trained model.
- `Requirement.txt`: List of required libraries.
- `training_data_file.txt`: Data file used to train the AutoRec model.
- `video.MOV` & `video.mp4`: Demonstration videos.
- `ml-1m`: Folder containing data files (`movies.dat`, `ratings.dat`, `users.dat`).
- `README.md`: This documentation file.

🚨 **Note**: Data preparation is embedded within `non_personalised.py` and `CF_personalised.py`.

## 📊 Data
Utilized Dataset: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
- `movies.dat`: Movie data file.
- `ratings.dat`: Ratings data file.
- `users.dat`: User data file.

## 📈 Results
Results may exhibit slight variations due to random data splitting for training and testing.
- **RMSE**: 
  - Collaborative Filtering: 1.08 
  - Non-Personalised: 1.33
- **nDCG10**: 
  - Collaborative Filtering: 0.10
  - Non-Personalised: 0.19
- **nDCG100**: 
  - Collaborative Filtering: 0.03
  - Non-Personalised: 0.14
- **HitRate(k=5)**: 
  - Collaborative Filtering: 0.22 
  - Non-Personalised: 0.23
- **HitRate(k=10)**: 
  - Collaborative Filtering: 0.11 
  - Non-Personalised: 0.23

## 📚 References
- Code: [AutoRec Reference](https://github.com/tuanio/AutoRec), [StackOverflow Text Color](https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python)
- Paper: [AutoRec Paper](http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf)
- Data: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

## 🛠 Training Parameters
- **Loss Function**: Adam
- **Learning Rate**: 0.0001
- **Batch Size**: 512
- **Epochs**: 100
- **Hidden Layers**: 500
