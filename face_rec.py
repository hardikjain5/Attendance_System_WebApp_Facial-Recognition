import numpy as np
import pandas as pd
import cv2
import os
import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

import time
from datetime import datetime

# Connect to Redis Client
hostname = 'redis-19498.c212.ap-south-1-1.ec2.cloud.redislabs.com'
portnumber = 19498
password = 'GWo29iVbFFsK7Lhq3GOGkQLuGUlQcbf2'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

# Retrieve data from database
def retrieve_data(name):
    retrieve_dict= r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df =  retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_role','facial_features']
    retrieve_df[['Name','Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name','Role','facial_features']]

# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,
                        name_role=['Name','Role'],thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
        
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    return person_name, person_role

### Real Time Prediction
# we need to save logs for every 1 min
class RealTimePred:
    def __init__(self):
        self.logs=dict(name=[],role=[],current_time=[])
    
    def reset_dict(self):
        self.logs=dict(name=[],role=[],current_time=[])

    def savelogs_redis(self):
        #step 1: create a logs dataframe
        dataframe=pd.DataFrame(self.logs)

        #step 2: drop the duplicate information(distinct name)
        dataframe.drop_duplicates('name',inplace=True)

        #step 3: push data to redis database(list)
        #encode the data
        name_list=dataframe['name'].tolist()
        role_list=dataframe['role'].tolist()
        ctime_list=dataframe['current_time'].tolist()
        encoded_data=[]

        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name!='Unknown':
                concat_list=f'{name}@{role}@{ctime}'
                encoded_data.append(concat_list)

        if len(encoded_data)>0:
            r.lpush('attendance:logs',*encoded_data)
        
        self.reset_dict()

    def face_predictions(self,test_image, dataframe,feature_column,
                        name_role=['Name','Role'],thresh=0.5):
    
        # time
        current_time=str(datetime.now())
        # step-1: take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        # step-2: use for loop and extract each embedding and pass to ml_search_algorithm

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            if person_name == 'Unknown':
                color =(0,0,255) # bgr
            else:
                color = (0,255,0)


            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.3,color,2)

            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

    # cv2.imshow('prediciton',test_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        return test_copy


# Registration Form

class RegistrationForm:
    def __init__(self):
        self.sample=0

    def reset(self):
        self.sample=0
    
    def get_embedding(self,frame):
          #get results from insightface model
          results=faceapp.get(frame,max_num=1)
          embeddings= None
          for res in results:
                
                self.sample+=1
                x1,y1,x2,y2=res['bbox'].astype(int)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
                #put smaples info into the frame
                text=f'samples= {self.sample}'
                cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)

                #extract facial features / facial embeddings
                embeddings=res['embedding']
                
          return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):

        #validate name before saving
        if name is not None:
            if name.strip(' ') != '':
                key=f'{name}@{role}'
            else:
                return 'name_false' 
        else:
            return 'name_false'

        # validate if face_embedding.txt is present or not
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'  

        #step 1: load the file "face embedding.txt file"
        x_array=np.loadtxt('face_embedding.txt',dtype=np.float32)  #convert into array


        #step 2: convert into array(into proper shape)
        received_samples=int(x_array.size/512)  #since each embedding is a vector of 512
        x_array=x_array.reshape(received_samples,512)
        x_array=np.asarray(x_array)

        #step 3: calculate mean embeddings
        x_mean=x_array.mean(axis=0)
        x_mean=x_mean.astype(np.float32)
        x_mean_bytes=x_mean.tobytes()

        #step 4: save in redis database.
        r.hset(name='academy:register',key=key,value=x_mean_bytes)

        os.remove('face_embedding.txt')
        self.reset()

        return True