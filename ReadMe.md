## Requirement library:
- pandas
- numpy
- scikit-learn
- torch
- matplotlib

## Usage:

- python main.py: run the implementation of the input interface and output interface. 
Note that the user_id range is from 1 to 6040.
The input interface will ask the user to input the user_id.

- python Evaluation.py: run the evaluation of the both model: AutoRec and Non-Personalised

- (REQUIRE uncomment some code)python CF_personalised.py: 
To test this main function, you need to uncomment the line 144 and comment the line 146.(down below the "def cf_personalised(user_id=100, num_recommendations=5)")
Then, you can run "python3/python CF_personalised.py" to test the AutoRec model with the trained model in the folder "model".

- (REQUIRE uncomment some code)python non_personalised.py: 
To test this main function, you need to uncomment the line 94 and comment the line 96. (down below the def non_personalised_rc(user_id=10, num_recommendations=10)
Then, you can run "python3/python non_personalised.py" to test the Non-Personalised model.

- (If you don't want to TRAIN/TEST the model, please ignore this file "AutoRecTraining.ipynb" and the following description)): 
Do run all to train the AutoRec model, and save the model in the folder "model"
If you run the train and model and save, the name will be "Autoencoder_valiate.pt".
Note: The training process will take a long time, so we have already trained the model and saved it in the folder "model".

## File description:
- main.py: The input interface and output interface. -- This is the main file to run
- Evalution.py: The evaluation of the both model: AutoRec and Non-Personalised
- AutoRecTraining.ipynb (This is no need for CW, unless want to test/train the model. Only evidence to show work): 
Training file. Do run all to train the AutoRec model, and save the model in the folder "model"
Note: There is trained model in the model file, and has already been implemented.
- CF_personalised.py: The implementation of the AutoRec model by using trained model in "model" folder, with evaluation function
- non_personalised.py: The implementation of the Non-Personalised model, with evaluation function
- bcolours.py: The file to print the text in the terminal with different color
- Evaluation_result.txt: The result of the evaluation of the both model: AutoRec and Non-Personalised
- model: The folder to save the trained model
- Requirement.txt: The requirement library
- training_data_file.txt: The training data file, which is the file to train the AutoRec model
- video.MOV: The video with MOV format.
- video.mp4: The video with MP4 format.
ml-1m: The folder to save the data file
- movies.dat: The data file of the movie
- ratings.dat: The data file of the rating
- users.dat: The data file of the user
- README.md: The file to describe the data

Note: data preparation is part of non_personalised.py and CF_personalised.py.

## Data:
MovieLens 1M Dataset: https://grouplens.org/datasets/movielens/1m/
- movies.dat: The data file of the movie
- ratings.dat: The data file of the rating
- users.dat: The data file of the user

## Result:
RMSE: Collaborative Filtering:  1.08  Non-Personalised:  1.33
nDCG10: Collaborative Filtering:  0.10  Non-Personalised:  0.19
nDCG100: Collaborative Filtering:  0.03  Non-Personalised:  0.14
HitRate(k=5): Collaborative Filtering:  0.22  Non-Personalised:  0.23
HitRate(k=10): Collaborative Filtering:  0.11  Non-Personalised:  0.23
## Note: The result would be different but similar if you run the code again, because the data is randomly split into training and testing data.

## Reference:
Code reference: https://github.com/tuanio/AutoRec
Code reference: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
Paper reference: http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf
Data reference: https://grouplens.org/datasets/movielens/1m/

## Training parameter: Adam as loss function, learning rate = 0.0001, batch size = 512, epoch = 100, hidden layer = 500
