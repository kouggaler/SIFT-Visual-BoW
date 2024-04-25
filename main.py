
import cv2
import pandas as pd
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
# evaluations
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import pickle
import os

###################################### Subroutines #####################################################################
"""
Example of subroutines you might need. 
You could add/modify your subroutines in this section. You can also delete the unnecessary functions.
It is encouraging but not necessary to name your subroutines as these examples. 
"""

def load_Imagefromfile(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR) 
    return image

def build_vocabulary(pathsegment, k):
    # train directory : pathsegment --> train/
    train_dirlist = [path for path in glob.glob(f"{pathsegment}/*")]
    sift = cv2.SIFT_create()
    des_arr = np.empty((0, 128))
    for directory in train_dirlist:
        imagepathlist =[path for path in glob.glob(f'{directory}/*')]
        for imagepath in imagepathlist:
            image = load_Imagefromfile(imagepath)
            keypoints, descriptors = sift.detectAndCompute(image, None)
            des_arr = np.vstack((des_arr, descriptors)) 

    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
    kmeans.fit(des_arr.astype('double'))
    codebook = kmeans.cluster_centers_

    codebook_dict = {}
    for index, vector in enumerate(codebook):
        codebook_dict[index] = vector
    codebook_df = pd.DataFrame.from_dict(codebook_dict, orient='index')

    codebook_df.to_csv(f"codebook_{k}.csv", index = False, header = codebook_df.columns.tolist())
    return codebook

def get_hist(pathsegment, codebook, k, n):
    # train directory : pathsegment --> train/
    # codebook
    train_dirlist = [path for path in glob.glob(f"{pathsegment}/*")]
    df_train_final = pd.DataFrame()
    for directory in train_dirlist:
        imagepathlist =[path for path in glob.glob(f'{directory}/*')]
        target = [directory.split("/")[-1].split("\\")[-1]]*len(imagepathlist)
        df_part = pd.DataFrame({
            'Image':imagepathlist,
            'Target':target
        })
        df_train_final= pd.concat([df_part, df_train_final], axis = 0)

    image_df = pd.DataFrame()
    sift = cv2.SIFT_create()
    for index, row in df_train_final.iterrows():
        image = load_Imagefromfile(row["Image"])
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        image_dict = {key: 0 for key in range(k)}
        for descriptor in descriptors:
            diff =  np.tile(descriptor, (k, 1)) - codebook  #k rows, 128 dim
            dists = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
            
            for i in range(n):
                image_dict[dists.argsort()[i]] += 1/dists[dists.argsort()[i]]

        heading = row["Target"]+"_"+str(index)
        image_df_part = pd.DataFrame.from_dict(image_dict, orient='index', columns = [heading] )
        image_df = image_df_part.merge(image_df, left_index=True, right_index=True, how = "outer")
    
    image_df.fillna(0, inplace=True)
    image_df_T = image_df.transpose()
    image_df_T= image_df_T.reset_index()
    image_df_T["target"] = image_df_T["index"].apply(lambda x : x.split("_")[0])
    image_train_T = image_df_T[image_df_T.columns[~image_df_T.columns.isin(["index"])]]

    return image_train_T
    

def classifier(image_train_T):
    # image_train_T
    labelsdict = {
                'TallBuilding': 15,
                'Suburb': 14,
                'Street': 13,
                'store': 12,
                'OpenCountry': 11,
                'Office': 10,
                'Mountain': 9,
                'livingroom': 8,
                'kitchen': 7,
                'Insidecity': 6,
                'industrial': 5,
                'Highway': 4,
                'Forest': 3,
                'Coast': 2,
                'bedroom': 1
                }
    
    num_cols = image_train_T.columns[~image_train_T.columns.isin(["target"])]
    preprocessor = ColumnTransformer([('num', StandardScaler(), num_cols)])
    X_train_prepared = preprocessor.fit_transform(image_train_T[num_cols])
    # X_train_prepared = image_train_T[num_cols]
    y_train = np.array(image_train_T["target"].apply(lambda x : labelsdict[x]))

    param_grid = {'kernel': ['rbf'],
                'degree' : [2],
                'gamma': ['scale'],
                'C': [ 2,3],
                "decision_function_shape" : ['ovo']
                }

    #class weight balance for auto scaling of weights based on inverse frequency
    svm_gs = GridSearchCV(estimator=SVC(class_weight='balanced'),
                        param_grid=param_grid, 
                        scoring='accuracy', 
                        cv=5
                        )

    svm_gs.fit(X_train_prepared, y_train)

    # print(svm_gs.best_params_)
    # print(svm_gs.best_score_)

    params = {}
    params['kernel'] = svm_gs.best_params_['kernel']
    params['gamma'] = svm_gs.best_params_['gamma'] 
    params['degree'] = svm_gs.best_params_['degree'] 
    params['C'] = svm_gs.best_params_['C']
    params["decision_function_shape"] = svm_gs.best_params_["decision_function_shape"]

    final_mdl = SVC(**params,class_weight='balanced')
    final_mdl.fit(X_train_prepared, y_train)
    
    return final_mdl, svm_gs.best_score_, preprocessor
    

def get_accuracy(y_test, y_test_pred):
    #y_test, y_test_pred
    return accuracy_score(y_test, y_test_pred)

def save_model(model, filename):
    # final mdl
    # "trained_svm.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""

def train(train_data_dir, model_dir):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    k = 150
    n = 5
    codebook = build_vocabulary(train_data_dir, k)
    image_train_T = get_hist(train_data_dir, codebook, k, n)
    model, accuracy, preprocessor = classifier(image_train_T)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_model(model, f"{model_dir}/trained_svm.pkl")
    save_model(preprocessor, f"{model_dir}/preprocessor.pkl")

    print(f"Train Accuracy : {accuracy:.3f}")
    return "Train End"


def test(test_data_dir, model_dir):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    k = 150
    n = 5

    test_dirlist = [path for path in glob.glob(f"{test_data_dir}/*")]

    df_test_final = pd.DataFrame()
    for directory in test_dirlist:
        imagepathlist =[path for path in glob.glob(f'{directory}/*')]
        target = [directory.split("/")[-1].split("\\")[-1]]*len(imagepathlist)
        df_part = pd.DataFrame({
            'Image':imagepathlist,
            'Target':target
        })
        df_test_final= pd.concat([df_part, df_test_final], axis = 0)

    sift = cv2.SIFT_create()
    image_df = pd.DataFrame()
    codebook = pd.read_csv(f"codebook_{k}.csv").values
    for index, row in df_test_final.iterrows():
        image = load_Imagefromfile(row["Image"])
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        image_dict = {key: 0 for key in range(k)}
        for descriptor in descriptors:
            diff =  np.tile(descriptor, (k, 1)) - codebook
            dists = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
            
            for i in range(n):
                image_dict[dists.argsort()[i]] += 1/dists[dists.argsort()[i]]
            
    #         predicted_cluster = dist.argsort()[0]
    #         image_dict[predicted_cluster] += 1

        heading = row["Target"]+"_"+str(index)
        image_df_part = pd.DataFrame.from_dict(image_dict, orient='index', columns = [heading] )
        image_df = image_df_part.merge(image_df, left_index=True, right_index=True, how = "outer")
        
    image_df.fillna(0, inplace=True)
    image_df_T = image_df.transpose()
    image_df_T= image_df_T.reset_index()
    image_df_T["target"] = image_df_T["index"].apply(lambda x : x.split("_")[0])
    image_test_T = image_df_T[image_df_T.columns[~image_df_T.columns.isin(["index"])]]

    labelsdict = {
            'TallBuilding': 15,
            'Suburb': 14,
            'Street': 13,
            'store': 12,
            'OpenCountry': 11,
            'Office': 10,
            'Mountain': 9,
            'livingroom': 8,
            'kitchen': 7,
            'Insidecity': 6,
            'industrial': 5,
            'Highway': 4,
            'Forest': 3,
            'Coast': 2,
            'bedroom': 1
            }
    num_cols = image_test_T.columns[~image_test_T.columns.isin(["target"])]

    with open(f"{model_dir}/preprocessor.pkl", 'rb') as file:
        preprocessor = pickle.load(file)

    X_test_prepared = preprocessor.transform(image_test_T[num_cols])
    # X_test_prepared = image_test_T[num_cols]
    y_test = np.array(image_test_T["target"].apply(lambda x : labelsdict[x]))

    with open(f"{model_dir}/trained_svm.pkl", 'rb') as file:
        model = pickle.load(file)
    
    y_test_pred = model.predict(X_test_prepared)
    accuracy = get_accuracy(y_test, y_test_pred)
    print(f"Test Accuracy : {accuracy:.3f}")
    return "Test End"



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test', help='the directory of testing data')
    parser.add_argument('--model_dir', default='./model', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)






