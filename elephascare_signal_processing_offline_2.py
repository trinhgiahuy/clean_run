import os
import sys

import psutil
if psutil.WINDOWS:
    sys.path.append(os.getcwd() + "\\src\\processing\\tools")
elif psutil.LINUX:
    sys.path.append(os.getcwd() + "/src/processing/tools")
import mysql.connector
from mysql.connector import Error
import bz2
import collections
import datetime
import gc
import json
import logging
import os
import os.path
import pickle
import queue
import sys
import threading
import time
import warnings
from datetime import datetime
from os import path, walk
from pickle import load
from collections import Counter
from scipy.signal.windows import blackmanharris
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import sorting
import tensorflow as tf
from sklearn.cluster import DBSCAN
import multiprocessing
import time

from tools.track_stats import Worker

if psutil.LINUX:
    processingPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/processing/tools"

# import torch
# from multiprocessing import Process

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
try:
    from parse_arguments import ParseArguments
    from store_data import StoreData
    from tools import Tools

except:
    from tools.parse_arguments import ParseArguments
    from tools.store_data import StoreData
    from tools.tools import Tools

from scipy import constants, signal
from scipy.stats import pearsonr
from termcolor import colored

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)
from memory_profiler import profile

# Global Static Variables
# ml_model_dir_path = os.getcwd() + "\\ML\\models\\LSTM_4class\\"
# # scaler = load(open(ml_model_dir_path + 'scaler.pkl', 'rb'))
# with open(ml_model_dir_path + 'scaler.pkl', 'rb') as file:
#     scaler = load(file)
# ml_model_path = ml_model_dir_path + "model_LSTM_4class.h5"
# model = tf.keras.models.load_model(ml_model_path)
'''
##########################################################################################################################################################################################################################################################################
1. Encapsulating into function and calling to maintain readability
# ENSURE WE HAVE TENSORFLOW FOR GPU COMPUTATIONS AS THAT CAN SPEED UP THE WORKLOAD
##########################################################################################################################################################################################################################################################################

'''
def load_model_and_scaler():
    if psutil.WINDOWS:
        ml_model_dir_path = os.getcwd() + "\\ML\\models\\LSTM_4class\\"
    elif psutil.LINUX:
        ml_model_dir_path = os.getcwd() + "/ml/models/LSTM_4class/"
    with open(ml_model_dir_path + 'scaler.pkl', 'rb') as file:
        scaler = load(file)
    ml_model_path = ml_model_dir_path + "model_LSTM_4class.h5"
    model = tf.keras.models.load_model(ml_model_path)
    return model, scaler

# Example of lazy loading when needed
model, scaler = load_model_and_scaler()
start_time_stamp = datetime.now()

# -----------------STATIC RADAR CONFIG-----------------
# NUM_CHIRPS = None
# CHIRP_SAMPLES = None
# MAX_SPEED_M_S = None
# MAX_RANGE_M = None
# RANGE_RESOLUTION_M = None
# MOVING_AVG_ALPHA = 0.6
# MTI_ALPHA = 1.0
# FALL_VEL_THRESHOLD = 0.5    def Bed_Detection(self, bed_cluster_in, Time_bed):
#
#         self.In_bed_previous = self.In_bed
#         all_data_points = [point for cluster in self.bed_points for point in cluster]
#         point_counts = Counter(map(tuple, all_data_points))
#         most_repeated_points = point_counts.most_common()
#         # print('>>>>>>>>>>>>>>>>>----------------->>>>>>>>>>>>>>>>point_counts',most_repeated_points)
#         # Assuming 'most_repeated_points' is a list of tuples containing (data_point, occurrence_count)
#         if not most_repeated_points:
#             # print(colored(
#             #     '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Not detected due to out of FoV '
#             #     '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
#             #     'red'))
#             logging.info(self.Time_bed + '  Status >>>>>> ' + " Not_detected")
#             self.In_bed = False
#
#             self.check_in_bed_transition
#
#         else:
#
#             # Extract x and y coordinates from the data points
#             most_repeated_cluster = most_repeated_points[0][
#                 0]  # Get the first data point with the highest occurrence count
#             most_repeated_cluster_set = set(most_repeated_cluster)  # Convert the most repeated cluster to a set
#              # Filter the points belonging to the most repeated cluster
#             most_repeated_cluster_points = [point for cluster in self.bed_points for point in cluster if
#                                             set(point) == most_repeated_cluster_set]
#
#             # Extract x and y coordinates from the most repeated cluster points
#             x_coords, y_coords = zip(*most_repeated_cluster_points)
#             XYbed = [x_coords, y_coords]
#
#             unique_bed_cluster_tuples = [data_point if isinstance(data_point, tuple) else (data_point,) for
#                                          data_point in XYbed]
#
#             # Convert the 'unique_bed_cluster_tuples' list to a NumPy array
#             unique_bed_cluster_array = np.array(unique_bed_cluster_tuples)
#
#
#             dist_to_bed = self.distance_to_cluster(np.unique(unique_bed_cluster_array), bed_cluster_in)
#
#
#             if dist_to_bed < 2:
#                 # print(colored(
#                 #     '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< In-bed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
#                 #     'yellow'))
#                 self.In_bed = True  # this should be analyzed if we are gonna check it over time or not
#                 # self.In_bed_list.append(1)
#
#                 self.Bed_Cnt += 1
#                 logging.info(self.Time_bed + '  Status >>>>>> ' + " In_bed")
#             else:
#
#                 self.In_bed = False
#                 # logging.info(self.Time_bed + '  Status >>>>>> ' + " Stationary")
#                 # self.In_bed = False
#                 # threading.Thread(
#                 #     name="bed_status_" + '_' + str(self.bbs),
#                 #     target=self.check_in_bed_transition,
#                 #     args=(),
#                 # ).start()
#                 # self.bbs += 1
#                 self.check_in_bed_transition
#
#             self.insert_into_bed_result_list(
#                 {'date': self.Time_bed, 'output': self.In_bed})
#
#             if (len(self.bed_result_list) >= 20):
#                 self.remove_from_bed_result_list()
#             self.bed_points = []
#
#         return
# MOVE_VEL_THRESHOLD = 0.2
# DIM = 2
# NUM_RX = 3
# ANGLELEN = NUM_CHIRPS * 2
# PHILEN = NUM_CHIRPS * 2

def create_database_and_table():
    try:
        # Connect to the MySQL server using provided credentials
        connection = mysql.connector.connect(
            host='localhost',
            user='root',  # MySQL username
            password='@2259586aA'  # MySQL password
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Create the specified database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS elephascare_ltc_edge_db")
            print("Database 'elephascare_ltc_edge_db' created or already exists.")

            # Select the newly created database
            cursor.execute("USE elephascare_ltc_edge_db")

            # Create the specified table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tbl_ml_tmp (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    output INT
                )
            """)
            print("Table 'tbl_ml_tmp' created or already exists.")

            cursor.close()
            connection.close()

    except Error as e:
        print(f"Error: {e}")


class ElephasCareSignalProcessing:
    def __init__(self, _STORAGE_LOCATION, _SENSOR_ADDRESS, _SERVER_INFO, _SENSOR_ID=1000):
        self.queue = queue.Queue()
        self.STORAGE_LOCATION = _STORAGE_LOCATION
        self.SENSOR_ADDRESS = _SENSOR_ADDRESS
        self.First_Run = False
        self.SERVER_INFO = _SERVER_INFO
        self.SENSOR_ID = _SENSOR_ID

        self.ml_rdm_queue = queue.Queue(maxsize=2048)
        self.fall_rdm_queue = queue.Queue(maxsize=2048)
        self.move_rdm_queue = queue.Queue(maxsize=2048)
        self.pad_rdm_queue = queue.Queue(maxsize=2048)
        self.suspected_fall_queue = queue.Queue(maxsize=2048)
        self.fall_final_queue = queue.Queue(maxsize=128)
        # self.suspected_move_queue = queue.Queue(maxsize=2048)
        self.move_final_queue = queue.Queue(maxsize=128)

        self.fall_result_dic = {}
        self.move_result_dic = {}
        self.pad_result_dic = {}

        self.ml_result_list = []
        self.bed_result_list = []

        self.RDM_List = {}
        # self.path1 = os.getcwd()+"\\src\\processing\\ML\\models\\LSTM_4class\\"   # this is for real-time run
        if psutil.WINDOWS:
            self.path1 = os.getcwd() + "\\ML\\models\\LSTM_4class\\"  # this is for offline run
        elif psutil.LINUX:
            self.path1 = os.getcwd() + "/ML/models/LSTM_4class/"  # this is for offline run

        try:
            # self.scaler = load(open(self.path1+'scaler.pkl', 'rb'))

            self.parallel_file_name = "parallel_" + self.SENSOR_ADDRESS + "_" + \
                                      str(datetime.now()).replace(':', '_').replace(
                                          ' ', '_').split(('.'))[0] + ".log"
            # this is for real-time run
            # logging.basicConfig(
            #     filename=os.getcwd()+"\\src\\processing\\logs\\"+self.parallel_file_name, level=logging.INFO)

            # this is for offline run
            if psutil.WINDOWS:
                logging.basicConfig(
                    filename=os.getcwd() + "\\logs\\" + self.parallel_file_name, level=logging.INFO)
            elif psutil.LINUX:
                logging.basicConfig(
                    filename=os.getcwd() + "/logs/" + self.parallel_file_name, level=logging.INFO)
        except:
            if psutil.WINDOWS:
                self.path1 = os.getcwd().split('\\commands')[0] + "\\ML\\models\\LSTM_4class\\"
            elif psutil.LINUX:
                self.path1 = os.getcwd().split('\\commands')[0] + "/ML/models/LSTM_4class/"
            # self.path1 = os.getcwd().split('\\commands')[0]+"\\ML\\models\\LSTM_4class\\"
            # self.scaler = load(open(self.path1+'scaler.pkl', 'rb'))

            self.parallel_file_name = "parallel_" + self.SENSOR_ADDRESS + "_" + \
                                      str(datetime.now()).replace(':', '_').replace(
                                          ' ', '_').split(('.'))[0] + ".log"
            if psutil.WINDOWS:
                logging.basicConfig(
                filename=os.getcwd().split('\\commands')[0] + "\\logs\\" + self.parallel_file_name, level=logging.INFO)
            elif psutil.LINUX:
                logging.basicConfig(
                filename=os.getcwd().split('/commands')[0] + "/logs/" + self.parallel_file_name, level=logging.INFO)

        # self.path = self.path1 + "model_LSTM_4class.h5"
        # self.path = self.path1 + "finalized_model_lstm_bedclass.joblib"
        # self.model = tf.keras.models.load_model(self.path)
        # self.model = joblib.load(self.path)

        # self.parallel_log=logging.
        self.parallel_file_name = "parallel_" + self.SENSOR_ADDRESS + "_" + \
                                  str(datetime.now()).replace(':', '_').replace(
                                      ' ', '_').split(('.'))[0] + ".log"
        # this is for real-time run
        # logging.basicConfig(
        #     filename=os.getcwd()+"\\src\\processing\\logs\\"+self.parallel_file_name, level=logging.INFO)

        # this is for offline run

        if psutil.WINDOWS:
            logging.basicConfig(
            filename=os.getcwd() + "\\logs\\" + self.parallel_file_name, level=logging.INFO)
        elif psutil.LINUX:
            logging.basicConfig(
            filename=os.getcwd() + "/logs/" + self.parallel_file_name, level=logging.INFO)
        # # logging.basicConfig(
        # # filename=os.getcwd()+"\\logs\\"+self.parallel_file_name, level=logging.INFO)

        # logging.info("Parallel Algorithm has Started...")

        self.tools = Tools()
        # self.db_info=tools.load_json_file(_SERVER_INFO)
        self.db_info = _SERVER_INFO

        self.Base_URL = 'http://172.16.4.110:8000/api/result/'
        return

    def add_into_queue(self, frame_name):
        self.queue.put(frame_name)
        return

    def remove_from_queue(self):
        self.queue.get()
        return

    def remove_from_queue_and_disk(self):
        frame_name = self.queue.pop()
        storedata = StoreData(self.STORAGE_LOCATION,
                              self.SENSOR_ADDRESS, frame_name, '')
        storedata.remove_from_disk()
        return

    def Capon_beaformer(self, Range_Dopp_prof):

        RngAzMat = np.zeros(
            (int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
        Rang_Prof_Az = np.zeros(
            (int(self.chirpsamples / 2), int(len(self.theta_vec)), int(self.dim)), dtype=complex)

        # # for azimuth
        Rang_Prof_Az[:, :, 0] = Range_Dopp_prof[0]
        Rang_Prof_Az[:, :, 1] = Range_Dopp_prof[2]
        del Range_Dopp_prof
        # RangeMatrix_her = 1 / self.numchirps * Rang_Prof_Az.conjugate().transpose([0, 2, 1])
        RangeMatrix_her = 1 / self.numchirps * np.conjugate(Rang_Prof_Az.transpose([0, 2, 1]))

        for rr in range(0, int(self.chirpsamples / 2)):
            #
            inv_R_hat = np.linalg.inv(RangeMatrix_her[rr, ...] @ Rang_Prof_Az[rr, ...])  # for azimuth

            for jj in range(len(self.theta_vec)):
                a_hat = np.array([1, self.a_thetta[jj]])
                self.y_spec[jj] = 1 / (a_hat.conjugate().transpose() @ inv_R_hat @ a_hat)

            RngAzMat[rr, :] = self.y_spec

        return RngAzMat

    def cfar_2d_before_lenovo(self, signal_matrix, guard_cells, training_cells, threshold_factor):
        num_rows, num_cols = signal_matrix.shape
        thresholded_matrix = np.zeros((num_rows, num_cols), dtype=np.int32)

        for i in range(num_rows - guard_cells - training_cells):
            for j in range(num_cols - guard_cells - training_cells):
                # Define the region of interest (ROI) based on guard cells and training cells
                row_start = max(0, i - guard_cells - training_cells)
                row_end = min(num_rows - 1, i + guard_cells + training_cells)
                col_start = max(0, j - guard_cells - training_cells)
                col_end = min(num_cols - 1, j + guard_cells + training_cells)

                # Calculate the local noise level (background) using training cells
                training_region = signal_matrix[row_start:row_end + 1, col_start:col_end + 1]
                training_sum = np.sum(training_region)
                num_training_cells = (row_end - row_start + 1) * (col_end - col_start + 1) - 1
                # noise_level = (training_sum - signal_matrix[i, j]) / num_training_cells
                noise_level = (training_sum) / num_training_cells

                # Calculate the threshold using the noise level and threshold factor
                threshold = noise_level * threshold_factor

                # Check if the signal is above the threshold (target detected)
                if signal_matrix[i, j] > threshold:
                    thresholded_matrix[i, j] = 1

        return thresholded_matrix

    def cfar_2d(self, signal_matrix, guard_cells, training_cells, threshold_factor):
        '''
        ##########################################################################################################################################################################################################################################################################
        4. Made use of slicing and mean
        Minor speed improvement.
        PLEASE CHECK IF CHANGE IS ACCURATE
        PLEASE CHECK IF CHANGE IS ACCURATE
        this is from sachin
        ##########################################################################################################################################################################################################################################################################

        '''
        num_rows, num_cols = signal_matrix.shape
        thresholded_matrix = np.zeros((num_rows, num_cols), dtype=np.int32)

        for i in range(num_rows - guard_cells - training_cells):
            for j in range(num_cols - guard_cells - training_cells):
                # Define the region of interest (ROI) using array slicing
                row_slice = slice(max(0, i - guard_cells - training_cells), min(num_rows - 1, i + guard_cells + training_cells + 1))
                col_slice = slice(max(0, j - guard_cells - training_cells), min(num_cols - 1, j + guard_cells + training_cells + 1))
                roi = signal_matrix[row_slice, col_slice]

                # Calculate the local noise level (background) using training cells
                noise_level = np.mean(roi)

                # Calculate the threshold using the noise level and threshold factor
                threshold = noise_level * threshold_factor

                # Check if the signal is above the threshold (target detected)
                if signal_matrix[i, j] > threshold:
                    thresholded_matrix[i, j] = 1

        return thresholded_matrix

    def Position_clustering(self, RngAzMat_sum, SENSOR_ADDRESS, date):

        thresholded_result = self.cfar_2d(np.abs(RngAzMat_sum), guard_cells=2, training_cells=8,
                                          threshold_factor=1.5)
        # Instantiate the DBSCAN model


        det_index = np.array(np.where(thresholded_result == 1))

        if det_index.shape[1] > 0:

            dbscan = DBSCAN(eps=1, min_samples=3, metric='euclidean')
            #
            X_dbscan = np.transpose(det_index)
            # # Fit the model to the data
            _ = dbscan.fit(X_dbscan)

            # # Retrieve the labels assigned to each data point
            labels = dbscan.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)

            if n_clusters_ > 1:
                # print('===============================multiple people %s ======================',SENSOR_ADDRESS)
                # print('===============================multiple people %s ======================',SENSOR_ADDRESS)
                # logging.info('  Counting >>>>>> ' + " multiple people" + SENSOR_ADDRESS)
                # logging.info(SENSOR_ADDRESS+" : "+date+' -> ' + " multiple people")
                i=0

            if n_clusters_ == 1:
                for k in unique_labels:
                    class_member_mask = labels == k
                    xy = X_dbscan[class_member_mask]
                    self.bed_points.append(xy)

        return self.bed_points

    def Bed_cluster_Generator(self):
        print(colored(
            '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Bed_cluster_Generator '
            '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
            'blue'))
        # print('>>>>>>>>>>>>>>>>>----------------->>>>>>>>>>>>>>>>self.bed_points', len(self.bed_points))

        all_data_points = [point for cluster in self.bed_points for point in cluster]
        point_counts = Counter(map(tuple, all_data_points))
        most_repeated_points = point_counts.most_common()
        # print('>>>>>>>>>>>>>>>>>----------------->>>>>>>>>>>>>>>>point_counts',most_repeated_points)
        # Assuming 'most_repeated_points' is a list of tuples containing (data_point, occurrence_count)
        if not most_repeated_points:
            print(colored(
                '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Not detected due to out of FoV '
                '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
                'red'))
            # print('>>>>>>>>>>>>>>>>>----------------->>>>>>>>>>>>>>>>self.bed_points', self.bed_points)

        else:
            self.Bd_C += 1
            print('Bd_C=====>', self.Bd_C)
            data_points, occurrence_counts = zip(*most_repeated_points)

            # Extract x and y coordinates from the data points
            # Create the scatter plot
            # plt.scatter(y_coords,x_coords , s=point_sizes, c='blue', alpha=0.5)

            most_repeated_cluster = most_repeated_points[0][
                0]  # Get the first data point with the highest occurrence count
            most_repeated_cluster_set = set(most_repeated_cluster)  # Convert the most repeated cluster to a set

            # Filter the points belonging to the most repeated cluster
            most_repeated_cluster_points = [point for cluster in self.bed_points for point in cluster if
                                            set(point) == most_repeated_cluster_set]

            # Extract x and y coordinates from the most repeated cluster points
            x_coords, y_coords = zip(*most_repeated_cluster_points)
            XYbed = [x_coords, y_coords]
            unique_bed_cluster_tuples = [data_point if isinstance(data_point, tuple) else (data_point,) for
                                         data_point in XYbed]

            # Convert the 'unique_bed_cluster_tuples' list to a NumPy array
            unique_bed_cluster_array = np.array(unique_bed_cluster_tuples)
            aaa = np.unique(unique_bed_cluster_array)
            a1 = aaa[0]
            a2 = aaa[1]

            self.bed_cluster.append(np.unique(unique_bed_cluster_array))  # adding to the bed_cluster
            np.save("bed_cluster3rdRIA.npy", self.bed_cluster)  # saving the bed_cluster
            # print('y_coords, x_coords', self.theta_vec[a2] * 180 / (np.pi), self.dist_points[a1])

            plt.figure(2)
            plt.scatter(y_coords, x_coords, c='blue', alpha=0.5)
            # Add labels and title
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.ylim(0, 64)
            plt.xlim(0, 256)

            # Show the plot
            plt.draw()
            plt.pause(1e-2)
        self.bed_points = []

        return

    def check_in_bed_transition(self):
        if psutil.LINUX:
            obj = Worker(logFilePath = str(processingPath)+"/logs/")
        # Check if self.In_bed transitioned from True to False
        if self.In_bed_previous and not self.In_bed:
            self.check_Pre_movement()
            # If the transition occurred, check the movement
            if self.Pre_Movment_detected:  # should implement movement check from the previous time to now, not just now
                self.In_bed = False
                self.Bed_Cnt = 0
                self.insert_into_bed_result_list(
                    {'date': self.Time_bed, 'output': self.In_bed})


            else:
                self.In_bed = True
                self.Bed_Cnt += 1
                self.insert_into_bed_result_list(
                    {'date': self.Time_bed, 'output': self.In_bed})
                # logging.info(self.Time_bed + '  Status >>>>>> ' + " In_bed")
                logging.info(self.SENSOR_ADDRESS+" : "+self.Time_bed + ' -> ' + " In_bed")
        
        if psutil.LINUX:
            obj.sendFinish()
        return

    def check_Pre_movement(self):

        index1 = self.find_ml_result_list_index2(
            self.Time_bed_previous)  # this might not give the correct result if the 2 dates are not equal (ahmadi)
        threshold_bed_transition = len(self.ml_result_list) - index1 - 1

        moving_counter = 0
        for i in range(index1, index1 + threshold_bed_transition):
            output = self.extract_output_from_ml_result_list(
                self.ml_result_list[i]['date'])
            if (str.lower(output) == 'moving'):
                moving_counter += 1

        if moving_counter > 6:
            self.Pre_Movment_detected = True
        else:
            self.Pre_Movment_detected = False
        # print('==================>moving_counter',moving_counter)
        # this index might change if half of the ml_result_list was removed, ahmadi
        #

        return

    def find_ml_result_list_index2(self, date):
        index = 0
        # print('len(self.ml_result_list)', self.ml_result_list[(len(self.ml_result_list)-1)]['date'] )

        for i in range(len(self.ml_result_list)):
            if self.ml_result_list[i]['date'] >= date:
                # print('self.ml_result_list[i][date]',self.ml_result_list[i]['date'])
                index = i
                return index
        return None

    def Bed_Detection(self, bed_cluster_in, Time_bed):
        if psutil.LINUX:
            obj = Worker(logFilePath = str(processingPath)+"/logs/")
        self.In_bed_previous = self.In_bed
        all_data_points = [point for cluster in self.bed_points for point in cluster]
        point_counts = Counter(map(tuple, all_data_points))
        most_repeated_points = point_counts.most_common()
        # del point_counts
        # del all_data_points
        # print('>>>>>>>>>>>>>>>>>----------------->>>>>>>>>>>>>>>>point_counts',most_repeated_points)
        # Assuming 'most_repeated_points' is a list of tuples containing (data_point, occurrence_count)
        if not most_repeated_points:
            # print(colored(
            #     '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Not detected due to out of FoV '
            #     '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
            #     'red'))
            # logging.info(self.Time_bed + '  Status >>>>>> ' + " Not_detected")
            logging.info(self.SENSOR_ADDRESS+" : "+self.Time_bed + ' -> ' + " Not_detected")
            self.In_bed = False
            # threading.Thread(
            #     name="bed_status_" + '_' + str(self.bbs),
            #     target=self.check_in_bed_transition,
            #     args=(),
            # ).start()
            self.check_in_bed_transition()
            self.bbs += 1
            # self.In_bed = False
            # self.In_bed_list.append(0)

        else:
            # data_points, occurrence_counts = zip(*most_repeated_points)

            # Extract x and y coordinates from the data points
            most_repeated_cluster = most_repeated_points[0][
                0]  # Get the first data point with the highest occurrence count
            most_repeated_cluster_set = set(most_repeated_cluster)  # Convert the most repeated cluster to a set
            # del most_repeated_cluster
            # Filter the points belonging to the most repeated cluster
            most_repeated_cluster_points = [point for cluster in self.bed_points for point in cluster if
                                            set(point) == most_repeated_cluster_set]

            # Extract x and y coordinates from the most repeated cluster points
            x_coords, y_coords = zip(*most_repeated_cluster_points)
            XYbed = [x_coords, y_coords]
            # del x_coords
            # del y_coords
            # del most_repeated_cluster_points
            unique_bed_cluster_tuples = [data_point if isinstance(data_point, tuple) else (data_point,) for
                                         data_point in XYbed]

            # Convert the 'unique_bed_cluster_tuples' list to a NumPy array
            unique_bed_cluster_array = np.array(unique_bed_cluster_tuples)
            # del unique_bed_cluster_tuples
            # del XYbed

            # bed_cluster.append(np.unique(unique_bed_cluster_array))  # adding to the bed_cluster
            # np.save("bed_cluster.npy", bed_cluster)  #saving the bed_cluster
            dist_to_bed = self.distance_to_cluster(np.unique(unique_bed_cluster_array), bed_cluster_in)
            #
            #
            # print('>>>>>>>>>>>>>>>>>----------------->>>>>>>>>>>>>>>>dist_to_bed', dist_to_bed)
            # print('>>>>>>>>>>>>>>>>>----------------->>>>>>>>>>>>>>>>bed_cluster', bed_cluster_in)
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>....position', np.unique(unique_bed_cluster_array))

            if dist_to_bed < 2:
                # print(colored(
                #     '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< In-bed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
                #     'yellow'))
                self.In_bed = True  # this should be analyzed if we are gonna check it over time or not
                # self.In_bed_list.append(1)

                self.Bed_Cnt += 1
                # logging.info(self.Time_bed + '  Status >>>>>> ' + " In_bed")
                logging.info(self.SENSOR_ADDRESS+" : "+self.Time_bed + ' -> ' + " In_bed")
            else:

                self.In_bed = False
                # logging.info(self.Time_bed + '  Status >>>>>> ' + " Stationary")
                # self.In_bed = False
                # threading.Thread(
                #     name="bed_status_" + '_' + str(self.bbs),
                #     target=self.check_in_bed_transition,
                #     args=(),
                # ).start()
                self.check_in_bed_transition()
                self.bbs += 1

            self.insert_into_bed_result_list(
                {'date': self.Time_bed, 'output': self.In_bed})

            if (len(self.bed_result_list) >= 20):
                self.remove_from_bed_result_list()
            self.bed_points = []

        if psutil.LINUX:
            obj.sendFinish()
        return

    def bed_exit(self, numchirps):
        # Count the number of True values in the 'output' key
        if self.In_bed == True and self.Bed_Cnt > 0 and self.MVE == True:
            # if self.MVE==True:
            print(colored(
                '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< bed movement is detected >>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
                'yellow'))
        return

    def entrance_exit(self, numchirps):
        # Count the number of True values in the 'output' key
        if self.In_bed == True and self.Bed_Cnt > 0 and self.MVE == True:
            # if self.MVE==True:
            print(colored(
                '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< bed movement is detected >>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
                'yellow'))
        return

    def bed_stat_check(self):
        # Count the number of True values in the 'output' key
        if self.In_bed == True and self.Bed_Cnt > 1 and self.MVE == True:
            # if self.MVE==True:
            print(colored(
                '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< bed movement is detected >>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
                'yellow'))
        return

    def euclidean_distance(self, point1, point2):
        #################### using NP.LINALG.NORM(POINT1-POINT2) INSTEAD
        return np.linalg.norm(point1 - point2)

    def distance_to_cluster(self, point, cluster):
        distances = [self.euclidean_distance(point, cluster_point) for cluster_point in cluster]
        return min(distances)

    def SO_CFAR(self, input_vector, ng, nt):
        '''
        The SO_CFAR function will take in a row vector of a given length and determine if there is an object at any
        of the given indexes. For each Cell Under Test the summation of the neighbouring cells are taken based on the
        training length value. The summation for the left and right side are taken and the one with the lowest value will
        be multiplied with the value of T to determine the threshold value. If the value of the Cell Under Test is greater
        than the threshold value then an object has been detected at that location.
        These values will be stored in a vector containing 0's and 1's and will be passed on for clustering.

        .. note::
        This is a core function for the following functions:

            - :func:`ndarray_CFAR <radar.RadarFcnLib.ndarray_CFAR>`
            - :func:`ndarray_CFAR_new <radar.RadarFcnLib.ndarray_CFAR_new>`
            - :func:`ndarray_CFAR_nested <radar.RadarFcnLib.ndarray_CFAR_nested>`

        :param matrix: input row vector that contains the square-law detector samples
        :param ng: the gap length at each side of CUTs
        :param nt: the half of the length of training window
        :param pfa_val: false-alarm rate
        :return: returning a vector containing zeros and ones corresponding to the presence of targets or their absence,
                respectively.
        '''

        # N = 2 * nt
        P_fa = 0.3

        input_width = input_vector.size

        # If input width is less than value then it will not satisfy the condition of having a constant N
        if input_width < 2 * ng + 2 * nt + 1:
            raise Exception(
                'The length of the input should be > 2*ng + 2*nt + 1')

        above_threshold = np.zeros(input_width)
        above_threshold[:ng + 1] = np.nan
        above_threshold[-ng - 1:] = np.nan

        # Index is the current Cell Under Test(CUT) and is only values between the guard band
        for testCell in range(ng + 1, input_width - ng - 1):

            # Covers the cases where the upper and lower reference window is the same
            if testCell - ng - nt >= 0 and testCell + ng + nt < input_width:
                lower_window = input_vector[testCell - ng - nt:testCell - ng]
                upper_window = input_vector[testCell + ng:testCell + ng + nt]

            # The reference window either partially exist or doesn't exist on the lower bound
            else:
                # Covers the case where partial reference window exist on left side of testCell
                # Need to increase the reference window on the right side
                if testCell - ng - nt < 0:
                    # samples_right = N - (testCell - ng)  # Determines number of samples on opposite side to compensate
                    lower_window = input_vector[0:testCell - ng]
                    upper_window = input_vector[testCell +
                                                ng:testCell + ng + nt]

                # Covers the case where partial reference window exist on right side of testCell
                elif testCell + ng + nt > input_width:
                    # Determines number of samples on opposite side to compensate for partial samples
                    # samples_left = N - (input_width - 1 - testCell - ng)
                    lower_window = input_vector[testCell -
                                                ng - nt:testCell - ng]
                    upper_window = input_vector[testCell + ng:input_width]

            n1 = lower_window.size
            n2 = upper_window.size
            n_thr = n1 + n2
            lower_sum = np.sum(lower_window)
            upper_sum = np.sum(upper_window)
            # del lower_window
            # del upper_window
            summation = np.divide((lower_sum + upper_sum), n_thr)
            alpha = n_thr * (((P_fa ** (np.divide(-1, n_thr)))) - 1)

            if input_vector[testCell] > np.multiply(summation, alpha):
                above_threshold[testCell] = 1

            del summation

        return above_threshold

    def frame_list(self, date, room_number, storage_path, sensor_name, time, Base_URL, client):
        extention_url = "datalist/"
        params = {
            "date": date,
            "room_number": room_number,
            "storage_path": storage_path.split(":")[0],
            "sensor_name": sensor_name,
            "time": time
        }
        url = Base_URL + extention_url
        frame_list = client.get(url, params=params, timeout=100).json()[
            'frame_list']

        return frame_list

    def download_frame(self, date, room_number, storage_path, sensor_name, time, frame_name, Base_URL, client):
        extention_url = "download/"
        path2 = frame_name.split(" ")[1]
        if psutil.WINDOWS:
            path1 = storage_path + "\\\\DATA_" + date + "_" + time + "\\\\Room" + \
                room_number + "-" + sensor_name + "\\\\" + frame_name.split(" ")[0]
            params = {
            "path1": path1.replace('\\\\', '\\'),
            "path2": path2
            }
        elif psutil.LINUX:
            path1 = storage_path + "/DATA_" + date + "_" + time + "/Room" + \
                room_number + "-" + sensor_name + "/" + frame_name.split(" ")[0]
            params = {
            "path1": path1,
            "path2": path2
            }
        

        # req = requests.get(url=Base_URL+extention_url,params=params)
        # headers = {'Content-Type': 'application/json'}
        responsed_data = client.get(
            Base_URL + extention_url, params=params, timeout=1000).content
        # print(path1+"\\"+path2)
        decompressed_data = bz2.decompress(responsed_data)
        pickle_data = pickle.loads(decompressed_data)

        return pickle_data

    def upload_result(self, url, RESULT_LOCATION, SENSOR_ADDRESS, file_name, payload):
        if psutil.WINDOWS:
            location = RESULT_LOCATION + "\\" + SENSOR_ADDRESS
        elif psutil.LINUX:
            location = RESULT_LOCATION + "/" + SENSOR_ADDRESS
        # file_name = str(file_name) + ".json"
        date = file_name.split(SENSOR_ADDRESS + "_")[1].split(' ')[0]
        time = file_name.split(SENSOR_ADDRESS + "_")[1].split(' ')[1]

        params = {'date': date, 'time': time, 'location': location}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url + 'upload_json/', json=json.dumps(payload), headers=headers, params=params)

        return

    def post_ml_result_to_server(self, output_id, description, date):
        url = self.Base_URL + "post_ml_detection/"
        params = {
            "sensorid": self.SENSOR_ID,
            "boolresult": output_id,
            "description": description,
            "date": date
        }
        response = requests.post(url, data=params)

        if response.status_code != 200:
            print("Failed to post data")

        return

    def extract_date_time_room_sensor_url(self, STORAGE_LOCATION, SENSOR_ADDRESS, Base_URL):
        storage_path = STORAGE_LOCATION.split("\\\\")[0]
        # print("STORAGE_LOCATION")
        # print(STORAGE_LOCATION)
        print(STORAGE_LOCATION)
        if psutil.WINDOWS:
            date = STORAGE_LOCATION.split("\\\\")[1].split("_")[1]
            time = STORAGE_LOCATION.split("\\\\")[1].split("_")[2]
        elif psutil.LINUX:
            date = STORAGE_LOCATION.split("/")[1].split("_")[1]
            time = STORAGE_LOCATION.split("/")[1].split("_")[2]
        # print(SENSOR_ADDRESS)
        room_number = SENSOR_ADDRESS.split("Room")[1].split("-")[0]
        sensor_name = SENSOR_ADDRESS.split("Room")[1].split("-")[1]
        client = httpx.Client()
        sorted_filenames = self.frame_list(
            date, room_number, storage_path, sensor_name, time, Base_URL, client)

        total_size = 0
        # for name, value in locals().items():
        #     # Skip module references and the function itself
        #     if not name.startswith('__') and not isinstance(value, type(sys)) and name != 'calculate_global_variables_size':
        #         size = sys.getsizeof(value)
        #         print(f"Size of {name} = {size} bytes")
        #         total_size += size
        print(f"Total size of all global variables: {total_size} bytes")
        return storage_path, date, time, room_number, sensor_name, client, sorted_filenames

    def extract_date_time_room_sensor_url2(self, STORAGE_LOCATION, SENSOR_ADDRESS, Base_URL):
        if psutil.WINDOWS:
            storage_path = STORAGE_LOCATION.split("\\\\")[0]
        elif psutil.LINUX:
            storage_path = STORAGE_LOCATION.split("/")[0]

        # print("STORAGE_LOCATION")
        # print(STORAGE_LOCATION)
        print(STORAGE_LOCATION)
        if psutil.WINDOWS:
            date = STORAGE_LOCATION.split("\\\\")[1].split("_")[1]
            time = STORAGE_LOCATION.split("\\\\")[1].split("_")[2]
        elif psutil.LINUX:
            date = STORAGE_LOCATION.split("/")[1].split("_")[1]
            time = STORAGE_LOCATION.split("/")[1].split("_")[2]
        # print(SENSOR_ADDRESS)
        room_number = SENSOR_ADDRESS.split("Room")[1].split("-")[0]
        sensor_name = SENSOR_ADDRESS.split("Room")[1].split("-")[1]
        client = httpx.Client()
        sorted_filenames = self.frame_list(
            date, room_number, storage_path, sensor_name, time, Base_URL, client)

        total_size = 0
        # for name, value in locals().items():
        #     # Skip module references and the function itself
        #     if not name.startswith('__') and not isinstance(value, type(sys)) and name != 'calculate_global_variables_size':
        #         size = sys.getsizeof(value)
        #         print(f"Size of {name} = {size} bytes")
        #         total_size += size
        print(f"Total size of all global variables: {total_size} bytes")
        return storage_path, date, time, room_number, sensor_name, client, sorted_filenames

    def insert_into_ml_result_list(self, dict):
        """
        ################################################################### 8. UPDATED TO USE ENUMERATE
        Insert data into the machine learning result list.

        Args:
            data_dict (dict): Dictionary containing data to be inserted.

        Returns:
            list: Updated machine learning result list.
        """
        index = len(self.ml_result_list)

        for i, entry in enumerate(self.ml_result_list):
            if entry['date'] > dict['date']:
                index = i
                break

        self.ml_result_list.insert(index, (dict))
        return self.ml_result_list

    def insert_into_bed_result_list(self, dict):
        index = len(self.bed_result_list)
        # Searching for the position
        for i, entry in enumerate(self.bed_result_list):
            if entry['date'] > dict['date']:
                index = i
                break

        # Inserting n in the list
        self.bed_result_list.insert(index, (dict))
        return self.bed_result_list

    def remove_from_ml_result_list(self):
        # for i in range(int(len(self.ml_result_list)/2)):
        #     del self.ml_result_list[0]
        self.ml_result_list = self.ml_result_list[len(self.ml_result_list) // 2:]

        # gc.collect()
        # threading.Thread(
        #     name="garbage" + '_' + str(self.gbc),
        #     target=self.gc_collect,
        #     args=(),
        # ).start()
        # self.gbc += 1
        return

    def remove_from_bed_result_list(self):
        # for i in range(int(len(self.bed_result_list)/2)):
        #     del self.bed_result_list[0]

        self.bed_result_list = self.bed_result_list[len(self.bed_result_list) // 2:]

        # gc.collect()
        # threading.Thread(
        #     name="garbage" + '_' + str(self.gbc),
        #     target=self.gc_collect,
        #     args=(),
        # ).start()
        # self.gbc += 1
        return

    def remove_file_from_cold_disk(self, file_name):
        while True:
            try:
                #os.remove(file_name)
                break
            except:
                pass
        return

    # def remove_file_from_cold_disk_modifiedgtgpt(self, file_name):
    #     # Close the file explicitly before removal
    #     with open(file_name, 'rb'):
    #         pass
    #     os.remove(file_name)
    #     return
    # @profile
    def run_ml(self, step_time, count_step_time, ml_rdm_queue_list):  # run_ml_model
        if psutil.LINUX:
            obj = Worker(logFilePath = str(processingPath)+"/logs/")
        # print('=======================>running ML')
        step_time = int(step_time)

        output = 'detection'
        output_id = -1

        RDM_vector = []
        dates = []
        # '''

        for i in range(0, len(ml_rdm_queue_list)):

            rdm_element = ml_rdm_queue_list[i]
            tmp_rdm = self.tools.convert_b2z_to_pickle(rdm_element)
            # threading.Thread(
            #     name="remove_file_from_cold_disk_" + '_' + str(self.rdsk),
            #     target=self.remove_file_from_cold_disk,
            #     args=(rdm_element,),
            # ).start()

            self.remove_file_from_cold_disk(rdm_element)
            self.rdsk += 1

            dates.append(rdm_element.split('ml_')[1].split('.pickle.bz2')[0])
            date1 = rdm_element.split('ml_')[1].split('.pickle.bz2')[0]
            # del rdm_element
            New_rng = (tmp_rdm[:, self.vel_itemindex_ml])
            RDM = abs(New_rng.reshape(New_rng.shape[0], New_rng.shape[2])) + 0
            # del New_rng
            # del tmp_rdm
            if count_step_time == step_time:

                output = self.Model_Test(RDM_vector)

                # ---------------------------reset---------------------------
                count_step_time = 0
                RDM_vector = []
                del RDM
            else:
                count_step_time = count_step_time + 1
                vector_tmp = RDM.reshape(RDM.shape[0] * RDM.shape[1])
                RDM_vector.append((vector_tmp))
                del vector_tmp
                del RDM

        del (RDM_vector)
        if output == 'Moving':
            # logging.info(date1 + " >>>>>>ML " + output)
            logging.info(self.SENSOR_ADDRESS+" : "+date1 +" -> " + output)
        del date1

        for date in dates:
            if (date != ''):
                # multithreading
                self.insert_into_ml_result_list(
                    {'date': date, 'output': output})
                # print(date+ '>>>>>>'+ output)

        dates = []

        if (len(self.ml_result_list) >= 5120):
            self.remove_from_ml_result_list()
            # gc.collect()
            # threading.Thread(
            #     name="garbage" + '_' + str(self.gbc),
            #     target=self.gc_collect,
            #     args=(),
            # ).start()

            self.gc_collect()

            self.gbc += 1
            # '''

        if psutil.LINUX:
            obj.sendFinish()
            
        return

    # @profile
    def Model_Test(self, x_test):
        # tf.profiler.experimental.start('C:\\ElephasCare_LTC\\LTC_Result\\Processing\\DATA_2023-06-16_1243', options=tf.profiler.experimental.ProfilerOptions(
        #     host_tracer_level=2,
        #     python_tracer_level=1,
        #     device_tracer_level=1
        # ))

        output = ''
        X_test = ((np.array(x_test)))
        X2 = X_test.reshape(1, X_test.shape[0] * X_test.shape[1])
        try:
            X2 = scaler.transform(X2)
        except:
            return output
        X_test = X2.reshape(1, X_test.shape[0], X_test.shape[1])

        old_stdout = sys.stdout
        # this is for real-time
        # sys.stdout = open(str(os.getcwd()) +
        #                   "\\src\\processing\\logs\\null", "w")

        # this is for offline
        if psutil.WINDOWS:
            sys.stdout = open(str(os.getcwd()) +
                          "\\logs\\null", "w")
        elif psutil.LINUX:
            sys.stdout = open(str(os.getcwd()) +
                          "/logs/null", "w")

        pred_result = model(X_test, training=False)

        # stop_time_stamp = datetime.now()

        sys.stdout = old_stdout

        y_pred = np.argmax(pred_result, axis=-1)
        del pred_result
        tf.keras.backend.clear_session()

        if y_pred == 0:
            output = 'Empty'
            output_id = 0
        elif y_pred == 1:
            output = 'Moving'
            output_id = 1
        elif y_pred == 2:
            output = 'Stationary'
            output_id = 2
        elif y_pred == 3:
            output = 'In_bed'
            output_id = 3
        elif y_pred == 4:
            output = 'on_floor'
            output_id = 4
        else:
            output = 'unknown'
        del (X_test)
        del (X2)
        del (y_pred)
        # gc.collect()
        # threading.Thread(
        #     name="garbage" + '_' + str(self.gbc),
        #     target=self.gc_collect,
        #     args=(),
        # ).start()
        # self.gbc += 1
        # tf.profiler.experimental.stop()
        # return output,output_id
        return output

    def put_on_queue(self, queue_name, data):
        if (queue_name == "ml_rdm_queue"):
            if (self.ml_rdm_queue.full()):
                self.ml_rdm_queue.get()
            self.ml_rdm_queue.put(data)
        elif (queue_name == "fall_rdm_queue"):
            if (self.fall_rdm_queue.full()):
                self.fall_rdm_queue.get()
            self.fall_rdm_queue.put(data)
        elif (queue_name == "pad_rdm_queue"):
            if (self.pad_rdm_queue.full()):
                self.pad_rdm_queue.get()
            self.pad_rdm_queue.put(data)
        return



    def mkdir(self, _path):
        if psutil.WINDOWS:
            dirs = _path.split("\\")
            __path = dirs[0] + "\\\\"
        elif psutil.LINUX:
            dirs = _path.split("/")
            __path = dirs[0] + "/"
        del dirs[0]
        for dir in dirs:
            if (dir != ''):
                if psutil.WINDOWS:
                    __path += dir + "\\\\"
                elif psutil.LINUX:
                    __path += dir + "/"
                if (path.exists(__path) != True):
                    os.mkdir(__path)
        return

    def store_RDM(self, RDM, SENSOR_ADDRESS, file_name, RESULT_STORAGE):
        if psutil.WINDOWS:
            tmp_path = RESULT_STORAGE.replace(
                '\\\\', '\\') + '\\' + SENSOR_ADDRESS.replace(
                '\\\\', '\\') + "\\RDM\\"
            tmp_path = tmp_path.replace('\\\\', '\\')
            if (path.exists(tmp_path) != True):
                self.mkdir(tmp_path)
            destenition_file_name = tmp_path + "\\" + file_name + ".pickle.bz2"
        elif psutil.LINUX:
            tmp_path = RESULT_STORAGE + '/' + SENSOR_ADDRESS+ "/RDM/"
            tmp_path = tmp_path
            if (path.exists(tmp_path) != True):
                self.mkdir(tmp_path)
            destenition_file_name = tmp_path + "/" + file_name + ".pickle.bz2"
        ofile = bz2.BZ2File(destenition_file_name, 'wb')
        pickle.dump(abs(RDM), ofile)
        ofile.close()
        return destenition_file_name

    def add_RDM_List(self, date, data):
        tmp2 = {}
        tmp2['data'] = data
        tmp2['pad_flag'] = True  # Pad   Function Read Data
        tmp2['fall_flag'] = False  # Fall  Function Read Data
        tmp2['ml_flag'] = False  # ML    Function Read Data
        self.RDM_List[date] = tmp2
        return


    def remove_Element_from_RDM_if_all_Functions_Used(self, element):
        ################################################### CAN USE  if all(self.RDM_List[element].values())
        #if ((self.RDM_List[element]['pad_flag'] == True) and (self.RDM_List[element]['fall_flag'] == True) and (self.RDM_List[element]['ml_flag'] == True)):
        if all(self.RDM_List[element].values()):
            del self.RDM_List[element]
        return

    def send_notification(self, alert_type, description, room, destenition):
        messaging = FcmUtils()
        registration_token = []
        # Ahmad iPhone
        # registration_token.append('dHF9gc_73jg:APA91bGAlFok2adJUK-_VgGFxtJJMOCTHoxbRe2pMv3BpjzVZy2Huo6lc2Dqwm1ZyiGJKNYKvnMcSS2UPkr-_g6W_Ng0NHjjSu8zHNlk7WYPPiArwMarxo9Xg3d91VnVVpmpNOrCLgrS')
        # Ahmad iPad
        # registration_token.append('cgw5djtX6ug:APA91bFfXeDcJSp2LUn4zbOyJCmmW-CDY103y5CeDOilW-puWPLWPvgJUvLbQe6U8KHl45ul3tRUy9IPgHAAVDeX9mguBlTyKeSkE-jAZ1gd_4VO7LXU3v2mnemsb4c1kgDW7vv4r5VN')

        # Hajar iPhone
        # registration_token = 'c2h8cFLalnY:APA91bHF0Vsx9WfDgv9PUzChelH9mU4QHJA60IibQ-V7KhIj3JoIlhcgBrLBqB4Bt9oJ8vObRGvHWYYx7xvoWWOGTmLnoqTGGM2A6E9rgW70qNT21XUqpz28Em2gT6hXGG4crPrv55bD'
        # GoldSentintel iPhone
        registration_token.append(
            'dIi-AXzW7IY:APA91bEhuEiVxU3SkfE3NkDiDftcSJZVuhiykK7O0ZQxo-_qTcUraUsXZ8erwJoiLSHVTRkkNJW0n5qGJAv5xrXrd2iPWMJReua6Qcxc6bLw8A6Bm-DiYok0rQ4AY9LYURY815F6-0AX')
        print(description)
        data = {
            'alert_type': str(alert_type),
            'description': str(description),
            'room': str(room),
            'date': str(datetime.now()),
        }
        for token in registration_token:
            messaging.send_to_token(token, str(description),
                                    room, data, 'emergency-alarm.mp3')

    def notification_managment(self, command):
        while True:
            if self.fall_final_queue.empty() == False:
                date = self.fall_final_queue.get()
                self.send_notification(
                    'fall', 'Fall happent at ' + date, 'GoldSentintel Office', [])

            else:
                time.sleep(0.101)
                break

        return

    def find_ml_result_list_index(self, date):
        index = 0

        for i in range(len(self.ml_result_list)):
            if self.ml_result_list[i]['date'] == date:
                index = i
                return index
        return None

    def find_bed_result_list_index(self, date):
        index = 0

        # filtered_list = [item for item in self.bed_result_list if
        #                  self.bed_result_list[:]['date'] >= date]
        # print('filtered_list====>',filtered_list)
        for i in range(len(self.bed_result_list)):
            if self.bed_result_list[i]['date'] >= date:
                index = i
                return index
            # Filter the list to include only items after the specific date

        return None

    def extract_output_from_ml_result_list(self, date):
        index = 0
        for i in range(len(self.ml_result_list)):
            if self.ml_result_list[i]['date'] == date:
                return self.ml_result_list[i]['output']
        return

    def final_decision_for_fall(self, start_date):
        if psutil.LINUX:
            obj = Worker(logFilePath = str(processingPath)+"/logs/")
        threshold_bed_stationary = 250
        back_threshold = 40
        walking_counter_threshold = 20

        start = time.time()
        # print('===================>',start_date)

        # threshold_in_bed_check = 5

        while True:
            if self.suspected_fall_queue.empty() == False:
                start = time.time()

                try:
                    date = self.suspected_fall_queue.get()  # gets the time when the suspected fall was detected
                    # print('date+5sec', date + 5)
                    # print('date = self.suspected_fall_queue.get()',date)
                    # print('self.suspected_fall_queue3', self.suspected_fall_queue.qsize())
                    final_decision = False

                    while final_decision == False:
                        try:

                            index_bed_sfd = self.find_bed_result_list_index(date)

                            index = self.find_ml_result_list_index(
                                date)  # this might not give the correct result if the 2 dates are not equal (ahmadi)
                            if index != None:
                                # this index might change if half of the ml_result_list was removed, ahmadi

                                # print(index)
                                # if (index + threshold_bed_stationary) <= len(self.ml_result_list): # this might not give the correct result if the result of ML is not ready (ahmadi)
                                try:
                                    if (self.ml_result_list[index + threshold_bed_stationary]['date']):
                                        walking_counter = 0
                                        for i in range(index, index + threshold_bed_stationary):
                                            output = self.extract_output_from_ml_result_list(
                                                self.ml_result_list[i]['date'])
                                            if (str.lower(output) == 'moving'):
                                                walking_counter += 1
                                        if (walking_counter <= walking_counter_threshold):
                                            if (index - back_threshold >= 0):
                                                walking_counter = 0
                                                for i in range(index - back_threshold, index + 1):
                                                    output = self.extract_output_from_ml_result_list(
                                                        self.ml_result_list[i]['date'])
                                                    if (str.lower(output) == 'moving'):
                                                        walking_counter += 1
                                                        if walking_counter == 1 and self.In_bed == False:
                                                            if index_bed_sfd != None:
                                                                final_decision = True
                                                                # os.system('color')
                                                                # print(colored(
                                                                #     '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Suspected Fall Approved >>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
                                                                #     'red'))
                                                                # logging.info(date + " >>>>>>  Suspected Fall Approved")
                                                                logging.info(self.SENSOR_ADDRESS+" : "+date +" -> " +"Suspected Fall Approved")

                                                                break
                                                            else:
                                                                # print('-----------waiting to get in bed report-----')
                                                                time.sleep(0.01)
                                                final_decision = True
                                                break
                                            else:
                                                final_decision = True
                                                break
                                        else:
                                            final_decision = True
                                            break
                                except:

                                    time.sleep(0.01)

                            else:

                                time.sleep(0.01)
                        except:
                            time.sleep(0.01)
                except:
                    time.sleep(0.01)
            else:
                end = time.time()
                if end - start > 10:
                    if psutil.LINUX:
                        obj.sendFinish()
                    break
                time.sleep(0.1)
        return

    def run_signal_processing_fall_detection(self, count_vel, det_count_thereshold, fall_rdm_queue_list):
        # print('=======================>running SFD')
        if psutil.LINUX:
            obj = Worker(logFilePath = str(processingPath)+"/logs/")
        first_run = True
        count_vel_thereshold = int(count_vel)
        for i in range(0, len(fall_rdm_queue_list)):

            if first_run == True:
                first_run = False
                # ----------------------------------------------------------------
                # ------------------- One Time Processing ------------------------
                # ----------------------------------------------------------------
                # fall_vel_threshold = 0.5
                count_vel = 0
                det_count = 0
                abnorm_count = 0
                vel_old = (np.ones((self.numchirps * 2, 1), dtype=complex)) * (np.random.rand((self.numchirps * 2)))
                vel_filt_old = vel_old[self.vel_itemindex_fall]
                # del vel_old
                vel_point_filt = self.vel_point[self.vel_itemindex_fall]
                RngDopp_sum = np.zeros(
                    (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            rdm_element = fall_rdm_queue_list[i]
            fft_2 = self.tools.convert_b2z_to_pickle(rdm_element)

            # threading.Thread(
            #     name="remove_file_from_cold_disk_" + '_' + str(self.rdsk),
            #     target=self.remove_file_from_cold_disk,
            #     args=(rdm_element,),
            # ).start()

            self.remove_file_from_cold_disk(rdm_element)

            self.rdsk += 1
            date = rdm_element.split('fall_')[1].split('.pickle.bz2')[0]
            # del rdm_element

            if count_vel == count_vel_thereshold - 1:

                RngDopp_sum = fft_2 + RngDopp_sum
                vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                vel_filt = vel_sum[self.vel_itemindex_fall]
                corr_fall, _ = pearsonr(abs(vel_filt), abs(vel_filt_old))
                # del vel_sum

                # --------------------------Fall Condition check----------------

                if corr_fall > 0.5:
                    ng = 4
                    nt = 8
                    det_vel_sum = self.SO_CFAR(vel_filt, ng, nt)
                    ind_CFAR_vel = np.argwhere(det_vel_sum == 1)
                    # del det_vel_sum
                    det_vel = (np.abs(vel_point_filt[ind_CFAR_vel]))
                    # del vel_point_filt
                    # del ind_CFAR_vel
                    filtind_2 = (np.argwhere(det_vel < 2.8))
                    if len(filtind_2) > 0:
                        abnorm_count = abnorm_count + 1
                    # del filtind_2

                # ---------------------reset---------------------------------
                vel_filt_old = vel_filt + 0
                count_vel = 0
                det_count = det_count + 1
                RngDopp_sum = np.zeros(
                    (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            else:
                count_vel = count_vel + 1
                RngDopp_sum = fft_2 + RngDopp_sum

            # del fft_2

            if det_count == det_count_thereshold:
                # if abnorm_count < 5 and abnorm_count > 0:
                if abnorm_count > 1:
                    # os.system('color')
                    # print(colored(
                    #     '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Suspected Fall >>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 'yellow'))
                    self.suspected_fall_queue.put(str(date))
                    # logging.info(date + " >>>>>> suspected fall")

                abnorm_count = 0
                det_count = 0

            # del date
        if psutil.LINUX:
            obj.sendFinish()
        return

    def run_signal_processing_move(self, count_move, det_count_thereshold,
                                   move_rdm_queue_list, function_name):
        # print('=======================>running SMD')
        if psutil.LINUX:
            obj = Worker(logFilePath = str(processingPath)+"/logs/")
        first_run = True
        det_count_thereshold = 50
        count_move_thereshold = int(count_move)
        for i in range(0, len(move_rdm_queue_list)):

            if first_run == True:
                first_run = False
                # ----------------------------------------------------------------
                # ------------------- One Time Processing global------------------------
                # ----------------------------------------------------------------
                count_vel = 0
                vel_filt_old = self.vel_point[self.vel_itemindex_move]
                vel_point_filt = self.vel_point[self.vel_itemindex_move]  # could be global

                # ----------------------------------------------------------------
                # ------------------- One Time Processing ------------------------
                # ----------------------------------------------------------------

            rdm_element = move_rdm_queue_list[i]
            fft_2 = self.tools.convert_b2z_to_pickle(rdm_element)

            # threading.Thread(
            #     name="remove_file_from_cold_disk_" + '_' + str(self.rdsk),
            #     target=self.remove_file_from_cold_disk,
            #     args=(rdm_element,),
            # ).start()

            self.remove_file_from_cold_disk(rdm_element)

            self.rdsk += 1
            date = rdm_element.split('move_')[1].split('.pickle.bz2')[0]
            del rdm_element

            RngDopp_sum = fft_2 + 0

            if count_vel == count_move_thereshold - 1:  # number of frames to be summed up

                RngDopp_sum = fft_2 + RngDopp_sum
                vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                vel_filt = vel_sum[self.vel_itemindex_move]
                # del vel_sum

                corr_move, _ = pearsonr(abs(vel_filt), abs(vel_filt_old))

                # --------------------------Movement Condition check----------------

                if corr_move > 0.5:
                    ng = 4
                    nt = 8
                    det_vel_sum = self.SO_CFAR(vel_filt, ng, nt)
                    ind_CFAR_vel = np.argwhere(det_vel_sum == 1)
                    # del det_vel_sum
                    det_vel = (np.abs(vel_point_filt[ind_CFAR_vel]))
                    # del ind_CFAR_vel
                    filtind_2 = (np.argwhere(det_vel < 2.8))
                    # del det_vel
                    if len(filtind_2) > 0:
                        self.move_Cnt += 1

                # ---------------------reset---------------------------------
                vel_filt_old = vel_filt + 0
                count_vel = 0
                self.mv_det_count = self.mv_det_count + 1

                RngDopp_sum = np.zeros(
                    (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            else:

                count_vel = count_vel + 1
                RngDopp_sum = fft_2 + RngDopp_sum

            # del fft_2

        move_rdm_queue_list = []
        if self.mv_det_count == det_count_thereshold:

            # if we need to detect over time, this should be changed and we should use
            # the bed_exit thread
            if self.In_bed == True and self.move_Cnt > 5 and self.Bed_Cnt > 3:
                # if MVE==True:
                # print(colored(
                #     '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Bed movement is detected >>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
                #     'red'))
                # logging.info(date + " >>>>>>>> Bed movement is detected")
                logging.info(self.SENSOR_ADDRESS+" : "+date +" -> " +"Bed_Movement")

            # else:
            # should decide when it should be reset for bed movement
            # move_Cnt=0

            self.mv_det_count = 0
            self.move_Cnt = 0
        # del date

        if psutil.LINUX:
            obj.sendFinish()
        return

    def log_variable_size(self):
        # print('____________________________Loooooooooooooooooooooog______________________________________')
        # logging.info('self.ml_rdm_queue'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.ml_rdm_queue)))
        # logging.info('self.fall_rdm_queue'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.fall_rdm_queue)))
        # logging.info('self.pad_rdm_queue'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.pad_rdm_queue)))
        # logging.info('self.suspected_fall_queue'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.suspected_fall_queue)))
        # logging.info('self.fall_final_queue'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.fall_final_queue)))

        # logging.info('self.fall_result_dic'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.fall_result_dic)))
        # logging.info('self.pad_result_dic'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.pad_result_dic)))
        # logging.info('self.ml_result_list'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.ml_result_list)))
        # logging.info('self.RDM_List'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.RDM_List)))
        # logging.info('self.fall_final_queue'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #              str(sys.getsizeof(self.fall_final_queue)))

        ml_rdm_queue = (sys.getsizeof(self.ml_rdm_queue))
        fall_rdm_queue = str(sys.getsizeof(self.fall_rdm_queue))
        pad_rdm_queue = str(sys.getsizeof(self.pad_rdm_queue))
        suspected_fall_queue = str(sys.getsizeof(self.suspected_fall_queue))
        fall_final_queue = str(sys.getsizeof(self.fall_final_queue))
        fall_result_dic = str(sys.getsizeof(self.fall_result_dic))
        pad_result_dic = str(sys.getsizeof(self.pad_result_dic))
        ml_result_list = str(sys.getsizeof(self.ml_result_list))
        RDM_List = str(sys.getsizeof(self.RDM_List))
        fall_final_queue = str(sys.getsizeof(self.fall_final_queue))
        # h = hpy()

        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
        #                          key=lambda x: -x[1])[:10]:
        #     # print("{:>30}: {:>8}".format(name, self.sizeof_fmt(size)))
        #     logging.info('self.fall_final_queue'+">>>>>>>>>>>>>>>>>>>>>>>>>>>>" +
        #                  str("{:>30}: {:>8}".format(name, self.sizeof_fmt(size))))

        return

    def extract_based_on_date(self, sorted_filenames, date):
        new_sorted_filenames = []
        print(date)
        i = 0
        for frame in sorted_filenames:
            ++i
            if i == (len(sorted_filenames)) / 2:
                break
            if (frame > date):
                new_sorted_filenames.append(frame)
                # print(frame)

        return new_sorted_filenames

    def stft_sig(self, fft_2):
        dopp_vel = ((np.abs(fft_2)).sum(axis=0))
        self.stft_const.append(dopp_vel)
        if len(self.stft_const) == 50:
            # print('self.(self.stft_const)==100',len(self.stft_const))
            STFT_sig = np.transpose(self.stft_const)

            plt.imshow(10 * np.log10(pow(np.abs(STFT_sig), 2)),
                       interpolation='sinc', origin='lower')
            # plt.imshow(10 * np.array(pow(np.abs(self.stft_const))),
            #            interpolation='sinc', origin='lower')
            # power = np.mean(pow(np.abs(Zxx), 2), 0)
            # plt.colorbar()
            # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=10, shading='gouraud')
            # plt.title('Spectrogram Magnitude')
            plt.ylabel('Frequency Samples')
            plt.xlabel('Time Samples')
            plt.draw()
            plt.pause(10)
            plt.clf()

            half_index = int(len(self.stft_const) // 2)
            self.stft_const = self.stft_const[half_index:2 * half_index]
            # print('self.(self.stft_const) after', len(self.stft_const))

        return

    def gc_collect(self):
        gc.collect()
        return

    def reset_thrd_val(self, mm, bb, i, j, bm, rs, rngazm):
        if mm > 2000:
            mm = 0
        if bb > 2000:
            bb = 0
        if i > 2000:
            i = 0
        if j > 2000:
            j = 0
        if bm > 2000:
            bm = 0
        if self.bbs > 2000:
            self.bbs = 0
        if rs > 2000:
            rs = 0
        if rngazm > 2000:
            rngazm = 0
        if self.gbc > 2000:
            self.gbc = 0
        if self.rdsk > 2000:
            self.rdsk = 0

    def visualize_Pattern(self, STORAGE_LOCATION, RESULT_ADDRESS, SENSOR_ADDRESS, Base_URL, STEP_TIME,
                          FALL_COUNT, COUNT_VEL, det_count_thereshold):
        # STORAGE_LOCATION = "C:\\DATA\\"
        # SENSOR_ADDRESS = "Room302_1-Bed"
        # RESULT_ADDRESS = "C:\\DATA\Room302_1-Bed"
        # Base_URL = "http://10.32.119.65:8000/api/datatransfer/"
        STEP_TIME = 4
        COUNT_VEL = 1
        COUNT_MOVE = 1
        det_count_thereshold = 10
        det_count_move = 10
        # --------------------------------Approving thread for fall------------------------------------

        # parrrallel_fall_approval = threading.Thread(
        #     name="Finall_Fall_Detection_" + SENSOR_ADDRESS,
        #     target=self.final_decision_for_fall,
        #     args=(datetime.now(),),
        # ).start()

        self.final_decision_for_fall(datetime.now())

        # first_run = True
        storage_path, date, time, room_number, sensor_name, client, sorted_filenames = self.extract_date_time_room_sensor_url(
            STORAGE_LOCATION, SENSOR_ADDRESS, Base_URL)

        first_run = True
        date2 = "RIA-Bed_2023-08-23 19-47-22.015757.pickle"
        # print("///////////////////////////////////////")
        # sorted_filenames = self.extract_based_on_date(sorted_filenames, date2)
        for frame in sorted_filenames:

            data = self.download_frame(
                date, room_number, storage_path, sensor_name, time, frame, Base_URL, client)
            # print('data["radar_config"]=====>', data["radar_config"]['date'])
            if first_run == True:
                first_run = False
                mm = 0
                bb = 0
                self.bbs = 0
                self.gbc = 0
                self.rdsk = 0
                i = 0
                j = 0
                bm = 0
                rs = 0
                rngazm = 0
                ml_rdm_queue_list = []
                fall_rdm_queue_list = []
                move_rdm_queue_list = []
                self.PAFlag = True
                self.In_bed = False
                self.In_bed_previous = False
                self.Pre_Movment_detected = False
                self.MVE = False
                self.move_Cnt = 0
                self.mv_det_count = 0
                self.Bd_C = 0  # if we want to create a bed cluster
                self.Bed_Cnt = 0
                self.stion_Cnt = 0
                self.Time_bed = str(data["radar_config"]['date'])
                bed_cluster_in = np.load("bed_cluster339.npy")
                # bed_cluster_in = np.load("bed_cluster3rdRIA.npy")

                # -----------------static radar config-----------------
                self.numchirps = data["radar_config"]['numchirps']
                self.chirpsamples = data["radar_config"]['chirpsamples']
                self.max_speed_m_s = data["radar_config"]['max_speed_m_s']
                self.max_range_m = data["radar_config"]['max_range_m']
                self.range_resolution_m = data["radar_config"]['range_resolution_m']

                # -----------------constants and threshold could be global-----------------
                moving_avg_alpha = 0.6
                mti_alpha = 1.0
                self.fall_vel_threshold = 0.5
                self.move_vel_threshold = 0.2
                self.dim = 2
                self.num_rx = 3
                self.anglelen = self.numchirps * 2
                self.philen = self.numchirps * 2

                # -----------------static vectors and matrices-----------------
                self.dist_points = np.linspace(
                    0, self.max_range_m / 2, int(self.chirpsamples / 2))  # comment it if you are not plotting anything

                self.vel_point = np.linspace(-self.max_speed_m_s,
                                             self.max_speed_m_s, self.numchirps * 2)
                self.theta_vec = np.linspace(-np.pi / 2, np.pi / 2, self.anglelen)
                self.y_spec = np.zeros(int(len(self.theta_vec)), dtype=complex)
                self.a_thetta = np.exp(-1j * np.pi * np.sin(self.theta_vec))

                self.vel_itemindex_move = np.where(
                    np.abs(np.array(self.vel_point)) > self.move_vel_threshold)
                self.vel_itemindex_fall = np.where(
                    np.abs(np.array(self.vel_point)) > self.fall_vel_threshold)
                vel_threshold_ml = 2
                self.vel_itemindex_ml = np.where(
                    np.abs(np.array(self.vel_point)) < vel_threshold_ml)
                # -----------------variables-----------------

                self.count_vel = 0
                self.final_count = 0
                self.absence_count = 0
                self.Present_count = 0
                self.abnorm_count = 0
                self.det_count = 0
                count_step_time = 0
                RngDopp_sum = np.zeros(
                    (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
                RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
                vel_old = np.zeros((self.numchirps * 2, 1), dtype=complex)
                Range_Dopp_prof = []
                self.bed_points = []
                self.bed_cluster = []

                # compute Blackman-Harris Window matrix over chirp samples(range)
                range_window = blackmanharris(
                    self.chirpsamples).reshape(1, self.chirpsamples)
                # compute Blackman-Harris Window matrix over number of chirps(velocity)
                doppler_window = blackmanharris(
                    self.numchirps).reshape(1, self.numchirps)

                # initialize doppler averages for all antennae
                dopp_avg = np.zeros(
                    (self.chirpsamples // 2, self.numchirps * 2, self.num_rx), dtype=complex)

            fft_2 = np.zeros(
                (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)

            for iAnt in range(0, 3):
                ele = "RX" + str(iAnt + 1)
                mat = data[ele]
                del ele

                # ----------------------------------------------------------------
                # ----------------------- Main Routine ---------------------------
                # ----------------------------------------------------------------
                # A loop for fetching and processing a finite number of frames
                avgs = np.average(mat, 1).reshape(self.numchirps, 1)

                # de-bias values
                mat = mat - avgs

                # -------------------------------------------------
                # Step 2 - Windowing the Data
                # -------------------------------------------------
                mat = np.multiply(mat, range_window)

                # -------------------------------------------------
                # Step 3 - add zero padding here
                # -------------------------------------------------
                zp1 = np.pad(
                    mat, ((0, 0), (0, self.chirpsamples)), 'constant')

                del mat
                del avgs

                # -------------------------------------------------
                # Step 4 - Compute FFT for distance information
                # -------------------------------------------------
                range_fft = np.fft.fft(zp1) / self.chirpsamples
                del zp1
                # ignore the redundant info in negative spectrum
                # compensate energy by doubling magnitude
                range_fft = 2 * range_fft[:,
                                range(int(self.chirpsamples / 2))]

                # # prepare for dopplerfft
                # ------------------------------------------------
                # Transpose
                # Distance is now indicated on y axis
                # ------------------------------------------------
                fft1d = range_fft + 0
                del range_fft
                # fft1d[0:skip] = 0
                fft1d = np.transpose(fft1d)

                # -------------------------------------------------
                # Step 7 - Windowing the Data in doppler
                # -------------------------------------------------
                fft1d = np.multiply(fft1d, doppler_window)

                zp2 = np.pad(
                    fft1d, ((0, 0), (0, self.numchirps)), 'constant')

                del fft1d
                fft2d = np.fft.fft(zp2) / self.numchirps
                del zp2
                # update moving average
                dopp_avg[:, :, iAnt] = (
                                               fft2d * moving_avg_alpha) + (
                                               dopp_avg[:, :, iAnt] * (1 - moving_avg_alpha))
                # MTI processing
                # needed to remove static objects
                # step 1 moving average
                # multiply history by (mti_alpha)
                # mti_alpha=0
                fft2d_mti = fft2d - (dopp_avg[:, :, iAnt] * mti_alpha)

                # re-arrange fft result for zero speed at centre
                dopplerfft = np.fft.fftshift(fft2d_mti, (1,))
                Range_Dopp_prof.append(dopplerfft)  # appending to range profile for azimuth and elevation
                fft_2 = fft_2 + dopplerfft  # integration over channels
                del fft2d
                del fft2d_mti
                del dopplerfft

            # '''

            # #--------------------------------ML thread for activity recognition------------------------------------

            ml_rdm_path = self.store_RDM(fft_2, SENSOR_ADDRESS, 'ml_' + data["radar_config"]['date'],
                                         RESULT_ADDRESS)

            ml_rdm_queue_list.append(ml_rdm_path)
            # print('ml_rdm_queue_list_before', len(ml_rdm_queue_list))

            if (len(ml_rdm_queue_list) == int(STEP_TIME + 1)):
                self.run_ml(STEP_TIME, count_step_time,
                          ml_rdm_queue_list)
                # threading.Thread(
                #     name="ML2_" + '_' + str(i),
                #     target=self.run_ml,
                #     args=(STEP_TIME, count_step_time,
                #           ml_rdm_queue_list),
                # ).start()
                i += 1
                ml_rdm_queue_list = []

            # '''
            # --------------------------------Suspected Fall detection-----------------------------------
            #
            fall_rdm_path = self.store_RDM(fft_2, SENSOR_ADDRESS, 'fall_' + data["radar_config"]['date'],
                                           RESULT_ADDRESS)
            fall_rdm_queue_list.append(fall_rdm_path)

            if (len(fall_rdm_queue_list) == int(COUNT_VEL) * int(det_count_thereshold)):
                self.run_signal_processing_fall_detection(COUNT_VEL, int(det_count_thereshold),
                          fall_rdm_queue_list)
                # fall_rdm_queue_list2=fall_rdm_queue_list
                # threading.Thread(
                #     name="Fall2_" + '_' + str(j),
                #     target=self.run_signal_processing_fall_detection,
                #     args=(COUNT_VEL, int(det_count_thereshold),
                #           fall_rdm_queue_list,),
                # ).start()

                fall_rdm_queue_list = []
                j += 1

            # --------------------------------suspected movement detection thread ------------------------------------

            move_rdm_path = self.store_RDM(fft_2, SENSOR_ADDRESS, 'move_' + data["radar_config"]['date'],
                                           RESULT_ADDRESS)

            move_rdm_queue_list.append(move_rdm_path)
            if (len(move_rdm_queue_list) == int(COUNT_MOVE) * int(det_count_move)):
                self.run_signal_processing_move(COUNT_MOVE, int(det_count_move),
                          move_rdm_queue_list, "move_" + '_' + str(mm))
                # threading.Thread(
                #     name="move_" + '_' + str(mm),
                #     target=self.run_signal_processing_move,
                #     args=(COUNT_MOVE, int(det_count_move),
                #           move_rdm_queue_list, "move_" + '_' + str(mm),),
                # ).start()
                move_rdm_queue_list = []
                mm += 1

            # --------------------------------resetting the threshold ------------------------------------
            self.reset_thrd_val(mm, bb, i, j, bm, rs, rngazm)
            # threading.Thread(
            #     name="reset_thrd_val" + '_' + str(rs),
            #     target=self.reset_thrd_val,
            #     args=(mm, bb, i, j, bm, rs, rngazm),
            # ).start()

            rs += 1

            if self.count_vel == 10:  ## number of frames,if count_vel == 10==> 1s

                RngDopp_sum = fft_2 + RngDopp_sum
                ### -------------------------------------------------
                # ------------------capon beam former + CFAR----------------
                ## -------------------------------------------------

                RngAzMat_sum = self.Capon_beaformer(Range_Dopp_prof) + RngAzMat_sum
                Range_Dopp_prof = []
                #
                self.Position_clustering(RngAzMat_sum, SENSOR_ADDRESS,data["radar_config"]['date'])
                # """

                ### -------------------------------------------------
                # ------------------Plotting RDM, RAM----------------
                ## -------------------------------------------------
                thresholded_result = self.cfar_2d(np.abs(RngAzMat_sum), guard_cells=2, training_cells=8,
                                                  threshold_factor=1.5)
                plt.figure(1)
                plt.clf()
                plt.cla()
                plt.subplot(311)
                ## plot3 = plt.figure("RngDoppMat")
                plt.imshow(abs(RngDopp_sum), cmap='hot',
                           extent=(-self.max_speed_m_s, self.max_speed_m_s,
                                   0, self.max_range_m / 2),
                           origin='lower')
                # cbar = plt.colorbar(h)
                plt.xlabel("velocity (m/s)")
                plt.ylabel("distance (m)")
                plt.title(data["radar_config"]['date'])
                plt.subplot(312)
                plt.pcolormesh(self.theta_vec * 180 / (np.pi), self.dist_points, thresholded_result, alpha=None, norm=None,
                               cmap=None, shading=None, antialiased=None)
                plt.subplot(313)
                plt.pcolormesh(self.theta_vec * 180 / (np.pi), self.dist_points, np.abs(RngAzMat_sum), alpha=None, norm=None,
                               cmap=None, shading=None, antialiased=None)

                plt.xlabel("Azimuth Angle [deg]")
                plt.ylabel("Range [m]")
                # plt.title(data["radar_config"]['date'])
                # plt.draw()
                plt.pause(1e-2)
                plt.draw()
                # print('self.In_bed',self.In_bed)
               # """

                ### -------------------------------------------------
                # ------------------ PAD algorithm----------------
                ## -------------------------------------------------

                vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                corr, _ = pearsonr(abs(vel_old), abs(vel_sum))
                if corr > 0.6:
                    # print('Present')
                    self.Present_count = self.Present_count + 1

                else:
                    self.absence_count = self.absence_count + 1

                # ---------------------reset---------------------------------
                vel_old = vel_sum + 0
                del vel_sum
                self.count_vel = 0
                self.final_count = self.final_count + 1
                RngDopp_sum = np.zeros(
                    (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
                RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)

            else:

                self.count_vel = self.count_vel + 1

                RngDopp_sum = fft_2 + RngDopp_sum
                # -------------------------------------capon beam former in another thread to expedite the process----------------
                # one of the time consuming processes is capon. how to solve it?
                RngAzMat_sum = self.Capon_beaformer(Range_Dopp_prof) + RngAzMat_sum

                Range_Dopp_prof = []

            if self.final_count == 10:  # making decision after 1 mins
                self.Time_bed_previous = self.Time_bed
                self.Time_bed = str(data["radar_config"]['date'])
                T_detct = self.Present_count + self.absence_count
                thr = self.Present_count / T_detct
                # logging.info('Duration= '+  str(datetime.now() - start_time_stamp))


                if thr > 0.7:
                    self.PAFlag = True
                    # print('Present',self.Time_bed)
                    # logging.info(str(data["radar_config"]['date']) + '  PAD >>>>>> ' + " Present" + ': '+SENSOR_ADDRESS)
                    logging.info(SENSOR_ADDRESS +" : "+str(data["radar_config"]['date']) + ' -> ' + " Present")
                    ####---adding the bed detection in another thread-------
                    self.Bed_Detection(bed_cluster_in, self.Time_bed)
                    # threading.Thread(
                    #     name="bed_" + '_' + str(bb),
                    #     target=self.Bed_Detection,
                    #     args=(bed_cluster_in, self.Time_bed),
                    # ).start()
                    bb += 1


                else:
                    # self.PAFlag = False
                    self.In_bed = False
                    # print('Absent')
                    # logging.info(str(data["radar_config"]['date']) + '  PAD >>>>>> ' + " Absent" + ': '+SENSOR_ADDRESS)
                    logging.info(SENSOR_ADDRESS +" : "+str(data["radar_config"]['date']) + ' -> ' + " Absent")
                    # threading.Thread(
                    #     name="bed_status_" + '_' + str(self.bbs),
                    #     target=self.check_in_bed_transition,
                    #     args=(),
                    # ).start()
                    # self.bbs += 1

                self.final_count = 0
                self.Present_count = 0
                self.absence_count = 0
                T_detct = 0
                # threading.Thread(
                #     name="garbage" + '_' + str(self.gbc),
                #     target=self.gc_collect,
                #     args=(),
                # ).start()
                # self.gbc += 1
            # # Get all threads
            # all_threads = threading.enumerate()
            # print(f"Number of non-daemon threads: {len(all_threads)}")
            # '''
            # print('Duration=', datetime.now()-start_time_stamp)

        return

    def get_sorted_files(self, path):
        """
        This function takes a file path and returns a list of file names in that path,
        sorted alphabetically. It filters out directories and only includes files.
        """
        try:
            # List all files and directories in the path
            files_and_dirs = os.listdir(path)

            # Filtering out directories and keeping only file names
            file_names = []

            for f in files_and_dirs:
                if os.path.isfile(os.path.join(path, f)):
                    file_names.append(f)
            
            #file_names = [f for f in files_and_dirs if os.path.isfile(os.path.join(path, f))]

            print(len(file_names))
            # Sorting the file names
            file_names.sort()

            return file_names
        except Exception as e:
            return str(e)
    def remove_prefix(self,input_file, output_file):
        """
        Removes the "INFO:root:" prefix from lines in the input log file and writes the modified lines to the output file.

        Args:
        input_file (str): Path to the input log file.
        output_file (str): Path to the output log file.

        Returns:
        None
        """
        # Open the input file for reading and the output file for writing
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Iterate through each line in the input file
            for line in infile:
                # Check if the line starts with the prefix "INFO:root:"
                if line.startswith("INFO:root:"):
                    # Remove the prefix from the line
                    modified_line = line[len("INFO:root:"):]
                else:
                    # If the line doesn't start with the prefix, keep it unchanged
                    modified_line = line
                
                # Write the modified line to the output file
                outfile.write(modified_line)
    def visualize_Pattern2(self, STORAGE_LOCATION, RESULT_ADDRESS, SENSOR_ADDRESS, Base_URL, STEP_TIME,
                           FALL_COUNT, COUNT_VEL, det_count_thereshold):
        
        if psutil.LINUX:
            obj = Worker(logFilePath = str(processingPath)+"/logs/")
        
        STEP_TIME = 4
        COUNT_VEL = 1
        COUNT_MOVE = 1
        det_count_thereshold = 10
        det_count_move = 10
        start_time_stamp=datetime.now()
        # --------------------------------Approving thread for fall------------------------------------
        self.final_decision_for_fall(datetime.now())
        # threading.Thread(
        #     name="Finall_Fall_Detection_" + SENSOR_ADDRESS,
        #     target=self.final_decision_for_fall,
        #     args=(datetime.now(),),
        # ).start()
        print("Please wait while the sorted filenames are loading..")
        if psutil.WINDOWS:
            sorted_filenames = self.get_sorted_files(STORAGE_LOCATION + "\\" + SENSOR_ADDRESS)
        elif psutil.LINUX:
            sorted_filenames = self.get_sorted_files(STORAGE_LOCATION)
        print(len(sorted_filenames))
        first_run = True
        # date2 = "Room339_1-Livingroom_2023-07-28 10-52-06.484328.pickle"
        # # print("///////////////////////////////////////")
        # sorted_filenames = self.extract_based_on_date(sorted_filenames, date2)
        for frame in sorted_filenames:
            #print(frame)
            storedata = StoreData(
                self.STORAGE_LOCATION, self.SENSOR_ADDRESS, frame, '')
            data = storedata.convert_pickle_bz2_2_json()
            frame_date=frame.split('.pickle')[0].split('_')[1]
            #print(frame_date)
            data["radar_config"]['date']=frame_date
            #print(data)
            if first_run == True:
                print("Processing has started..")
                first_run = False
                mm = 0
                bb = 0
                self.bbs = 0
                self.gbc = 0
                self.rdsk = 0
                i = 0
                j = 0
                bm = 0
                rs = 0
                rngazm = 0
                ml_rdm_queue_list = []
                fall_rdm_queue_list = []
                move_rdm_queue_list = []
                self.PAFlag = True
                self.In_bed = False
                self.In_bed_previous = False
                self.Pre_Movment_detected = False
                self.MVE = False
                self.move_Cnt = 0
                self.mv_det_count = 0
                self.Bd_C = 0  # if we want to create a bed cluster
                self.Bed_Cnt = 0
                self.stion_Cnt = 0
                self.Time_bed = str(data["radar_config"]['date'])
                # bed_cluster_in = np.load("bed_cluster339.npy")
                bed_cluster_in = np.load("bed_clusterRIA.npy")

                # -----------------static radar config-----------------
                self.numchirps = data["radar_config"]['numchirps']
                self.chirpsamples = data["radar_config"]['chirpsamples']
                self.max_speed_m_s = data["radar_config"]['max_speed_m_s']
                self.max_range_m = data["radar_config"]['max_range_m']
                self.range_resolution_m = data["radar_config"]['range_resolution_m']

                # -----------------constants and threshold could be global-----------------
                moving_avg_alpha = 0.6
                mti_alpha = 1.0
                self.fall_vel_threshold = 0.5
                self.move_vel_threshold = 0.2
                self.dim = 2
                self.num_rx = 3
                self.anglelen = self.numchirps * 2
                self.philen = self.numchirps * 2

                # -----------------static vectors and matrices-----------------
                self.dist_points = np.linspace(
                    0, self.max_range_m / 2, int(self.chirpsamples / 2))  # comment it if you are not plotting anything

                self.vel_point = np.linspace(-self.max_speed_m_s,
                                             self.max_speed_m_s, self.numchirps * 2)
                self.theta_vec = np.linspace(-np.pi / 2, np.pi / 2, self.anglelen)
                self.y_spec = np.zeros(int(len(self.theta_vec)), dtype=complex)
                self.a_thetta = np.exp(-1j * np.pi * np.sin(self.theta_vec))

                self.vel_itemindex_move = np.where(
                    np.abs(np.array(self.vel_point)) > self.move_vel_threshold)
                self.vel_itemindex_fall = np.where(
                    np.abs(np.array(self.vel_point)) > self.fall_vel_threshold)
                vel_threshold_ml = 2
                self.vel_itemindex_ml = np.where(
                    np.abs(np.array(self.vel_point)) < vel_threshold_ml)
                # -----------------variables-----------------

                self.count_vel = 0
                self.final_count = 0
                self.absence_count = 0
                self.Present_count = 0
                self.abnorm_count = 0
                self.det_count = 0
                count_step_time = 0
                RngDopp_sum = np.zeros(
                    (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
                RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
                vel_old = np.zeros((self.numchirps * 2, 1), dtype=complex)
                Range_Dopp_prof = []
                self.bed_points = []
                self.bed_cluster = []

                # compute Blackman-Harris Window matrix over chirp samples(range)
                range_window = blackmanharris(
                    self.chirpsamples).reshape(1, self.chirpsamples)
                # compute Blackman-Harris Window matrix over number of chirps(velocity)
                doppler_window = blackmanharris(
                    self.numchirps).reshape(1, self.numchirps)

                # initialize doppler averages for all antennae
                dopp_avg = np.zeros(
                    (self.chirpsamples // 2, self.numchirps * 2, self.num_rx), dtype=complex)

            fft_2 = np.zeros(
                (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)

            for iAnt in range(0, 3):
                ele = "RX" + str(iAnt + 1)
                mat = data[ele]
                # print('============================================================================>')
                # print(mat)
                # print('============================================================================>')
                #
                # plt.pause(100e-1)



                del ele

                # ----------------------------------------------------------------
                # ----------------------- Main Routine ---------------------------
                # ----------------------------------------------------------------
                # A loop for fetching and processing a finite number of frames
                avgs = np.average(mat, 1).reshape(self.numchirps, 1)

                # de-bias values
                mat = mat - avgs


                # -------------------------------------------------
                # Step 2 - Windowing the Data
                # -------------------------------------------------
                mat = np.multiply(mat, range_window)

                # -------------------------------------------------
                # Step 3 - add zero padding here
                # -------------------------------------------------
                zp1 = np.pad(
                    mat, ((0, 0), (0, self.chirpsamples)), 'constant')

                del mat
                del avgs

                # -------------------------------------------------
                # Step 4 - Compute FFT for distance information
                # -------------------------------------------------
                range_fft = np.fft.fft(zp1) / self.chirpsamples
                del zp1
                # ignore the redundant info in negative spectrum
                # compensate energy by doubling magnitude
                range_fft = 2 * range_fft[:,
                                range(int(self.chirpsamples / 2))]

                # # prepare for dopplerfft
                # ------------------------------------------------
                # Transpose
                # Distance is now indicated on y axis
                # ------------------------------------------------
                fft1d = range_fft + 0
                del range_fft
                # fft1d[0:skip] = 0
                fft1d = np.transpose(fft1d)

                # -------------------------------------------------
                # Step 7 - Windowing the Data in doppler
                # -------------------------------------------------
                fft1d = np.multiply(fft1d, doppler_window)

                zp2 = np.pad(
                    fft1d, ((0, 0), (0, self.numchirps)), 'constant')

                del fft1d
                fft2d = np.fft.fft(zp2) / self.numchirps
                del zp2
                # update moving average
                dopp_avg[:, :, iAnt] = (
                                               fft2d * moving_avg_alpha) + (
                                               dopp_avg[:, :, iAnt] * (1 - moving_avg_alpha))
                # MTI processing
                # needed to remove static objects
                # step 1 moving average
                # multiply history by (mti_alpha)
                # mti_alpha=0
                fft2d_mti = fft2d - (dopp_avg[:, :, iAnt] * mti_alpha)

                # re-arrange fft result for zero speed at centre
                dopplerfft = np.fft.fftshift(fft2d_mti, (1,))
                Range_Dopp_prof.append(dopplerfft)  # appending to range profile for azimuth and elevation
                fft_2 = fft_2 + dopplerfft  # integration over channels

                del fft2d
                del fft2d_mti
                del dopplerfft


            # '''

            # #--------------------------------ML thread for activity recognition------------------------------------

            ml_rdm_path = self.store_RDM(fft_2, SENSOR_ADDRESS, 'ml_' + data["radar_config"]['date'],
                                         RESULT_ADDRESS)

            ml_rdm_queue_list.append(ml_rdm_path)
            # print('ml_rdm_queue_list_before', len(ml_rdm_queue_list))

            if (len(ml_rdm_queue_list) == int(STEP_TIME + 1)):
                self.run_ml(STEP_TIME, count_step_time,
                          ml_rdm_queue_list)
                # threading.Thread(
                #     name="ML2_" + '_' + str(i),
                #     target=self.run_ml,
                #     args=(STEP_TIME, count_step_time,
                #           ml_rdm_queue_list),
                # ).start()
                i += 1
                ml_rdm_queue_list = []

            # '''
            # --------------------------------Suspected Fall detection-----------------------------------
            #
            fall_rdm_path = self.store_RDM(fft_2, SENSOR_ADDRESS, 'fall_' + data["radar_config"]['date'],
                                           RESULT_ADDRESS)
            fall_rdm_queue_list.append(fall_rdm_path)

            if (len(fall_rdm_queue_list) == int(COUNT_VEL) * int(det_count_thereshold)):
                # fall_rdm_queue_list2=fall_rdm_queue_list
                self.run_signal_processing_fall_detection(COUNT_VEL, int(det_count_thereshold),
                          fall_rdm_queue_list)
                # threading.Thread(
                #     name="Fall2_" + '_' + str(j),
                #     target=self.run_signal_processing_fall_detection,
                #     args=(COUNT_VEL, int(det_count_thereshold),
                #           fall_rdm_queue_list,),
                # ).start()

                fall_rdm_queue_list = []
                j += 1

            # --------------------------------suspected movement detection thread ------------------------------------

            move_rdm_path = self.store_RDM(fft_2, SENSOR_ADDRESS, 'move_' + data["radar_config"]['date'],
                                           RESULT_ADDRESS)

            move_rdm_queue_list.append(move_rdm_path)
            if (len(move_rdm_queue_list) == int(COUNT_MOVE) * int(det_count_move)):
                self.run_signal_processing_move(COUNT_MOVE, int(det_count_move),
                          move_rdm_queue_list, "move_" + '_' + str(mm))
                # threading.Thread(
                #     name="move_" + '_' + str(mm),
                #     target=self.run_signal_processing_move,
                #     args=(COUNT_MOVE, int(det_count_move),
                #           move_rdm_queue_list, "move_" + '_' + str(mm),),
                # ).start()
                move_rdm_queue_list = []
                mm += 1

            # --------------------------------resetting the threshold ------------------------------------
            self.reset_thrd_val(mm, bb, i, j, bm, rs, rngazm)
            # threading.Thread(
            #     name="reset_thrd_val" + '_' + str(rs),
            #     target=self.reset_thrd_val,
            #     args=(mm, bb, i, j, bm, rs, rngazm),
            # ).start()

            rs += 1

            if self.count_vel == 10:  ## number of frames,if count_vel == 10==> 1s

                RngDopp_sum = fft_2 + RngDopp_sum
                ### -------------------------------------------------
                # ------------------capon beam former + CFAR----------------
                ## -------------------------------------------------

                RngAzMat_sum = self.Capon_beaformer(Range_Dopp_prof) + RngAzMat_sum
                Range_Dopp_prof = []
                #
                self.Position_clustering(RngAzMat_sum, SENSOR_ADDRESS,data["radar_config"]['date'])
                # """

                ### -------------------------------------------------
                # ------------------Plotting RDM, RAM----------------
                ## -------------------------------------------------
                thresholded_result = self.cfar_2d(np.abs(RngAzMat_sum), guard_cells=2, training_cells=8,
                                                  threshold_factor=1.5)
                plt.figure(1)
                plt.clf()
                plt.cla()
                # plt.subplot(311)
                # ## plot3 = plt.figure("RngDoppMat")
                # plt.imshow(abs(RngDopp_sum), cmap='hot',
                #            extent=(-self.max_speed_m_s, self.max_speed_m_s,
                #                    0, self.max_range_m / 2),
                #            origin='lower')
                # # cbar = plt.colorbar(h)
                # plt.xlabel("velocity (m/s)")
                # plt.ylabel("distance (m)")
                # plt.title(data["radar_config"]['date'])
                # plt.subplot(312)
                # plt.pcolormesh(self.theta_vec * 180 / (np.pi), self.dist_points, thresholded_result, alpha=None, norm=None,
                #                cmap=None, shading=None, antialiased=None)
                # plt.subplot(313)
                plt.pcolormesh(self.theta_vec *( 180 / (np.pi)), self.dist_points, np.abs(RngAzMat_sum), alpha=None, norm=None,
                               cmap=None, shading=None, antialiased=None)


                plt.xlabel("Azimuth Angle [deg]")
                plt.ylabel("Range [m]")
                # plt.title(data["radar_config"]['date'])
                # plt.draw()
                plt.pause(1e-2)
                plt.draw()
                # print('self.In_bed',self.In_bed)
               # """

                ### -------------------------------------------------
                # ------------------ PAD algorithm----------------
                ## -------------------------------------------------

                vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                corr, _ = pearsonr(abs(vel_old), abs(vel_sum))
                if corr > 0.6:
                    # print('Present')
                    self.Present_count = self.Present_count + 1

                else:
                    self.absence_count = self.absence_count + 1

                # ---------------------reset---------------------------------
                vel_old = vel_sum + 0
                del vel_sum
                self.count_vel = 0
                self.final_count = self.final_count + 1
                RngDopp_sum = np.zeros(
                    (int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
                RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)

            else:

                self.count_vel = self.count_vel + 1

                RngDopp_sum = fft_2 + RngDopp_sum
                # -------------------------------------capon beam former in another thread to expedite the process----------------
                # one of the time consuming processes is capon. how to solve it?
                RngAzMat_sum = self.Capon_beaformer(Range_Dopp_prof) + RngAzMat_sum

                Range_Dopp_prof = []

            if self.final_count == 10:  # making decision after 1 mins
                self.Time_bed_previous = self.Time_bed
                self.Time_bed = str(data["radar_config"]['date'])
                T_detct = self.Present_count + self.absence_count
                thr = self.Present_count / T_detct
                info = 'Duration= ' + str(datetime.now() - start_time_stamp)
                threading.Thread(
                    name="Duration" + '_' + str(self.Time_bed),
                    target=logging.info,
                    args=(info,),
                ).start()
                if thr > 0.7:
                    self.PAFlag = True
                    # print('Present',self.Time_bed)
                    # logging.info(str(data["radar_config"]['date']) + '  PAD >>>>>> ' + " Present" + SENSOR_ADDRESS)
                    logging.info(SENSOR_ADDRESS +" : "+ str(data["radar_config"]['date']) + ' ->PAD ' + " Present")
                    ####---adding the bed detection in another thread-------'
                    self.Bed_Detection(bed_cluster_in, self.Time_bed)
                    # threading.Thread(
                    #     name="bed_" + '_' + str(bb),
                    #     target=self.Bed_Detection,
                    #     args=(bed_cluster_in, self.Time_bed),
                    # ).start()
                    bb += 1


                else:
                    # self.PAFlag = False
                    self.In_bed = False
                    # print('Absent')
                    # logging.info(str(data["radar_config"]['date']) + '  PAD >>>>>> ' + " Absent" + SENSOR_ADDRESS)
                    logging.info(SENSOR_ADDRESS +" : "+ str(data["radar_config"]['date']) + ' ->PAD ' + " Absent")

                    # threading.Thread(
                    #     name="bed_status_" + '_' + str(self.bbs),
                    #     target=self.check_in_bed_transition,
                    #     args=(),
                    # ).start()
                    # self.bbs += 1

                self.final_count = 0
                self.Present_count = 0
                self.absence_count = 0
                T_detct = 0
                start_time_stamp = datetime.now()
                # threading.Thread(
                #     name="garbage" + '_' + str(self.gbc),
                #     target=self.gc_collect,
                #     args=(),
                # ).start()
                # self.gbc += 1
            # # Get all threads
            # all_threads = threading.enumerate()
            # print(f"Number of non-daemon threads: {len(all_threads)}")
            # '''
        print("done")
        if psutil.LINUX:
            obj.sendFinish()

        print(gc.garbage)
        print(gc.isenabled())
        print(gc.get_stats())
        return

    def get_date(self, _list):
        return _list.get('date')

    def plot_log(self, log_path):
        """
        This function creates a timeline plot from a log file.

        Args:
        log_path (str): Path of the log file to be plotted.

        Returns:
        None

        Example:
        obj = class_name()
        obj.plot_log("/path/to/logfile.log")
        """
        ' >>>>>> '
        with open(log_path) as f:
            lines = f.readlines()
        new_lines = []
        new_lines_dict = []
        dates = []
        detection_label = []
        for line in lines:
            tmp_line = line.split('INFO:root:')[1].split(' >>>>>> ')
            if (len(tmp_line) >= 2):
                tmp_line_dic = {}
                tmp_line_dic['date'] = np.array(tmp_line[0])
                tmp_line_dic['detection_label'] = tmp_line[1].split('\n')[0]
                if (tmp_line_dic['detection_label'] == 'Empty'):
                    tmp_line_dic['detection_label'] = 'E'
                elif (tmp_line_dic['detection_label'] == 'Walking'):
                    tmp_line_dic['detection_label'] = 'W'
                elif (tmp_line_dic['detection_label'] == 'Stationary'):
                    tmp_line_dic['detection_label'] = 'S'
                elif (tmp_line_dic['detection_label'] == 'In_bed'):
                    tmp_line_dic['detection_label'] = 'B'
                elif (tmp_line_dic['detection_label'] == 'errrrrrrrrrrrrrror'):
                    tmp_line_dic['detection_label'] = ''
                elif (tmp_line_dic['detection_label'] == 'Suspected Fall Approved'):
                    tmp_line_dic['detection_label'] = 'SPFA'
                elif (tmp_line_dic['detection_label'] == 'suspected fall'):
                    tmp_line_dic['detection_label'] = 'SPF'

                new_lines_dict.append(tmp_line_dic)
                tmp_line = line.split('INFO:root:')[1].split(' >>>>>> ')

                # logging.info(line.split('INFO:root:')[1])

                self

        new_lines_dict.sort(key=self.get_date)
        dataframe = pd.DataFrame()
        dates = []
        data_frame = []
        current_detection = ''
        pre_detection = new_lines_dict[0]['detection_label']

        pre_date = new_lines_dict[0]['date']
        current_date = new_lines_dict[0]['date']
        for line in new_lines_dict:
            current_detection = line['detection_label']
            current_date = line['date']
            if (pre_detection != current_detection):
                tmp = {}
                tmp['detection_label'] = pre_detection
                tmp['start'] = str(pre_date).split(
                    ' ')[0] + ' ' + str(pre_date).split(' ')[1].replace('-', ':')

                tmp['end'] = str(current_date).split(
                    ' ')[0] + ' ' + str(current_date).split(' ')[1].replace('-', ':')

                pre_detection = current_detection
                pre_date = current_date
                data_frame.append(tmp)

                logging.info(tmp['start'])

        source = pd.DataFrame(data_frame)
        source['start'] = pd.to_datetime(source['start'])
        source['end'] = pd.to_datetime(source['end'])

        fig = px.timeline(source.sort_values('start'),
                          x_start="start",
                          x_end="end",
                          y="detection_label",
                          #   text="detection_label",
                          color_discrete_sequence=["green", "red", "blue", "goldenrod", "magenta", "yellow", "purple"])

        fig.show()

        return


def create_path_if_not_exist(path):
    """
    This function takes a file path and creates the directory if it does not already exist.
    """
    try:
        # Check if the path exists
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
            return f"Path '{path}' was created."
        else:
            return f"Path '{path}' already exists."
    except Exception as e:
        return str(e)


# Example usage:
# path = '/your/new/path'
# print(create_path_if_not_exist(path))

# main function of the ElephasCare Signal Processing library
def main():
    if psutil.LINUX:
        with open(processingPath + "/logs/log.log", "w") as f:
            f.write("")

        if os.path.exists(processingPath + "/logs"):
            for files in os.listdir(processingPath + "/logs"):
                if os.path.isfile(processingPath + "/logs/"+files) and "_log" in files:
                    os.remove(processingPath + "/logs/"+files)


    # create_database_and_table()
    # set the console color
    os.system('color')

    # print the ElephasCare header in green color
    print(colored("******************* ElephasCare     *******************", 'green'))
    print(colored("******************* Runinng Offline.. *******************", 'green'))

    # get arguments from the user input
    pars_args = ParseArguments()
    args = pars_args.get_server_arguments()

    # record start time
    start_time_stamp = datetime.now()

    # assign values to the following variables
    STORAGE_LOCATION = args.storage_location
    SENSOR_ADDRESS = args.sensor_address.split('+')
    RESULT_ADDRESS = args.result_address

    create_path_if_not_exist(RESULT_ADDRESS)

    process_type = args.run_type
    instance_count = args.instance_count

    # get the current working directory and parent directory paths
    path1 = os.getcwd()
    path2 = os.path.abspath(os.path.join(path1, os.pardir))
    path22 = os.path.abspath(os.path.join(path2, os.pardir))
    path3 = os.path.abspath(os.path.join(path22, os.pardir))

    # # initialize ElephasCareSignalProcessing class
    # offline_ecsp = ElephasCareSignalProcessing(
    #     STORAGE_LOCATION, SENSOR_ADDRESS[0], path3+"\\server_side\\info\\db_info.json")
    # create the result directory if it does not exist
    print(RESULT_ADDRESS)
    if (path.exists(RESULT_ADDRESS) != True):
        offline_ecsp.mkdir(RESULT_ADDRESS)
    # perform different operations based on the process type
    if (process_type == "local"):
        offline_ecsp.run_offline_presence_detection(RESULT_ADDRESS)
    elif (process_type == "network"):
        # run presence detection over network
        print("network...")
        Base_URL = args.base_url
        RESULT_BASE_URL = args.result_base_url
        offline_ecsp.run_offline_presence_detection_over_network(
            STORAGE_LOCATION, SENSOR_ADDRESS[0], RESULT_ADDRESS, Base_URL, RESULT_BASE_URL)
    elif (process_type == "ml_over_network"):
        # run machine learning model over network
        print("ml_over_network...")
        Base_URL = args.base_url
        print(Base_URL)
        offline_ecsp.run_ml_network(STORAGE_LOCATION, SENSOR_ADDRESS[0], Base_URL)
    elif (process_type == "parallel"):
        # run parallel processing
        print("parallel...")
        Base_URL = args.base_url
        print(Base_URL)
        STEP_TIME = args.step_time
        FALL_COUNT = args.fall_count
        COUNT_VEL = args.count_vel
        det_count_thereshold = args.detection_count_thereshold

        offline_ecsp.run_parallel_processing(
            STORAGE_LOCATION, RESULT_ADDRESS, SENSOR_ADDRESS[0], Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            det_count_thereshold)
    elif (process_type == "multilayer"):
        print("multilayer...")
        # Base_URL = sys.argv[5]
        # print(Base_URL)

        STEP_TIME = args.step_time
        FALL_COUNT = args.fall_count
        COUNT_VEL = args.count_vel
        det_count_thereshold = args.detection_count_thereshold
        Base_URL = args.base_url
        offline_ecsp.run_multilayer_processing(
            STORAGE_LOCATION, RESULT_ADDRESS, SENSOR_ADDRESS[0], Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            det_count_thereshold)
    elif (process_type == "visualize"):

        STEP_TIME = args.step_time
        FALL_COUNT = args.fall_count
        COUNT_VEL = args.count_vel
        det_count_thereshold = args.detection_count_thereshold
        Base_URL = args.base_url
        offline_ecsp.run_offline_for_visualize(
            STORAGE_LOCATION, RESULT_ADDRESS, SENSOR_ADDRESS[0], Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            det_count_thereshold)
    elif (process_type == "visualize_rdm"):
        STEP_TIME = args.step_time
        FALL_COUNT = args.fall_count
        COUNT_VEL = args.count_vel
        det_count_thereshold = args.detection_count_thereshold
        Base_URL = args.base_url
        offline_ecsp.visualize_RDM(
            STORAGE_LOCATION, RESULT_ADDRESS, SENSOR_ADDRESS[0], Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            det_count_thereshold)
    elif (process_type == "visualize_Pattern"):
        print("-----------------------")
        # initialize ElephasCareSignalProcessing class
        port=8000
        for sensor in SENSOR_ADDRESS:
            if psutil.WINDOWS:
                offline_ecsp = ElephasCareSignalProcessing(
                STORAGE_LOCATION, sensor, path3 + "\\server_side\\info\\db_info.json")
            elif psutil.LINUX:
                offline_ecsp = ElephasCareSignalProcessing(
                STORAGE_LOCATION, sensor, path3 + "/server_side/info/db_info.json")
            STEP_TIME = args.step_time
            FALL_COUNT = args.fall_count
            COUNT_VEL = args.count_vel
            det_count_thereshold = args.detection_count_thereshold
            Base_URL = args.base_url
            Base_URL=Base_URL.replace('8000',str(port))
            
            
            port=port+1
            print(Base_URL,str(port))
            offline_ecsp.visualize_Pattern(STORAGE_LOCATION, RESULT_ADDRESS + sensor, sensor, Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
                    det_count_thereshold)
            # threading.Thread(
            #     name="offline_ecsp_" + '_' + sensor,
            #     target=offline_ecsp.visualize_Pattern,
            #     args=(
            #         STORAGE_LOCATION, RESULT_ADDRESS + sensor, sensor, Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            #         det_count_thereshold,),
            # ).start()

            # offline_ecsp.visualize_Pattern(
            #     STORAGE_LOCATION, RESULT_ADDRESS+sensor, sensor, Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            #     det_count_thereshold)

    elif (process_type == "visualize_Pattern2"):
        print("-----------------------")
        # initialize ElephasCareSignalProcessing class
        for sensor in SENSOR_ADDRESS:
            if psutil.WINDOWS:
                offline_ecsp = ElephasCareSignalProcessing(
                    STORAGE_LOCATION, sensor, path3 + "\\info\\db_info.json")
            elif psutil.LINUX:
                offline_ecsp = ElephasCareSignalProcessing(
                    STORAGE_LOCATION, sensor, path3 + "/info/db_info.json")
            STEP_TIME = args.step_time
            FALL_COUNT = args.fall_count
            COUNT_VEL = args.count_vel
            det_count_thereshold = args.detection_count_thereshold
            Base_URL = args.base_url

            offline_ecsp.visualize_Pattern2(
                STORAGE_LOCATION, RESULT_ADDRESS+sensor, sensor, Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
                det_count_thereshold)

            # threading.Thread(
            #     name="offline_ecsp_" + '_' + sensor,
            #     target=offline_ecsp.visualize_Pattern2,
            #     args=(
            #         STORAGE_LOCATION, RESULT_ADDRESS + sensor, sensor, Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            #         det_count_thereshold,),
            # ).start()
            # multiprocessing.Process(
            #     name="offline_ecsp_" + '_' + sensor,
            #     target=offline_ecsp.visualize_Pattern,
            #     args=(
            #         STORAGE_LOCATION, RESULT_ADDRESS + sensor, sensor, Base_URL, STEP_TIME, FALL_COUNT, COUNT_VEL,
            #         det_count_thereshold,),
            # ).start()


    elif (process_type == "log_path"):
        print(str(args.log_path))
        if psutil.WINDOWS:
            log_path = str(os.getcwd()) + '\\' + str(args.log_path)
        if psutil.LINUX:
            log_path = str(os.getcwd()) + '/' + str(args.log_path)
        offline_ecsp.plot_log(log_path=log_path)
    stop_time_stamp = datetime.now()
    print("Start time:" + str(start_time_stamp))
    print("Stop time:" + str(stop_time_stamp))

if __name__ == '__main__':
    main()