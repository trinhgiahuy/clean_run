import bz2
import pickle
import glob
import threading
import queue
import numpy as np
import os
import os.path
import psutil
import logging
import tensorflow as tf
from sklearn.cluster import DBSCAN
from collections import Counter, deque
from collections import Counter
import gc
from memory_profiler import memory_usage
from scipy.signal.windows import blackmanharris
from datetime import datetime
from scipy.stats import pearsonr
import sys
from termcolor import colored
from pickle import load
from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor
# Huy's monitoring framework
# from src.processing.tools.my_track_stats import Worker

import time
import onnx
import onnxruntime as ort

# if psutil.LINUX:
#     processingPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/SE455-Lenovo-Server-Code/src/processing/tools"

# print(f"processingPath: {processingPath}")

# def load_model_and_scaler():
#     if os.name == 'nt':  # Windows
#         base_dir = os.path.join(os.getcwd(), "ML", "models", "LSTM_4class")
#     else:  # Linux or other OS
#         base_dir = os.path.join(os.getcwd(), "src", "processing", "ML", "models", "LSTM_4class")
    
#     # Construct the full paths
#     scaler_path = os.path.join(base_dir, "scaler.pkl")
#     model_path = os.path.join(base_dir, "model_LSTM_4class.h5")
        
#     # Load the scaler
#     with open(scaler_path, 'rb') as file:
#         scaler = load(file)
    
#     # Load the model
#     model = tf.keras.models.load_model(model_path)
    
#     return model, scaler

# model, scaler = load_model_and_scaler()

def load_model_and_scaler():
    base_dir = os.path.join(os.getcwd(), "src", "processing", "ML", "models", "LSTM_4class")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    onnx_model_path = "/home/h3trinh/clean_run/src/processing/ML/models/LSTM_4class/test2.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(onnx_model_path)

    with open(scaler_path, 'rb') as file:
        scaler = load(file)
    
    return scaler, ort_session

scaler, ort_session = load_model_and_scaler()




class ProducerUpdate:

    def __init__(self, dataPath: str, numThreads: int = 0):
        self.start_time_stamp = datetime.now()
        self.queue = queue.Queue()
        self.suspected_fall_queue = queue.Queue(maxsize=2048)
        # self.ml_result_list = []
        # self.bed_result_list = []
        self.ml_result_list = deque(maxlen=5120)  # Use deque for memory efficiency
        self.bed_result_list = deque(maxlen=20)   # Smaller deque for bed result
        self.In_bed = False

        self.dataPath = dataPath
        self.First_Run = False
        # Justin;s data comprise of 7000 files
        # *51 equivalent to 10 hours

        # 1 day = 86400 seconds
        # 12 hours = 43200 seconds = 432000 frames so multiply by ~62 to get 12 hours
        self.pickleList = glob.glob(self.dataPath)*62*2
        self.files = []
        self.processed_files = 0
        self.number_of_files: int = len(self.pickleList)

        self.unzip_event = threading.Event()

        # Use ThreadPoolExecutor instead of creating new threads
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.unzip_thread = threading.Thread(target=self.unzip_files)
        self.unzip_thread.start()

        self.run_benchmarking_thread = threading.Thread(
            name="Run_benchmark",
            target=self.run_benchmark2,
            args=(self.queue, self.number_of_files)
        )
        self.run_benchmarking_thread.start()

        self.unzip_thread.join()
        self.unzip_event.set()
        self.run_benchmarking_thread.join()

    def unzip_files(self):
        for file in self.pickleList:
            self.unzip_file(file)

    def unzip_file(self, fileName: str):
        with bz2.BZ2File(fileName, 'rb') as ifile:
            pickle_data = pickle.load(ifile)
            self.queue.put(pickle_data)
            logging.info(f"Added file {fileName} to queue. Current queue size: {self.queue.qsize()}")

    def initialize_parameters(self, data):
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
    
        self.numchirps = data["radar_config"]['numchirps']
        self.chirpsamples = data["radar_config"]['chirpsamples']
        self.max_speed_m_s = data["radar_config"]['max_speed_m_s']
        self.max_range_m = data["radar_config"]['max_range_m']
        self.range_resolution_m = data["radar_config"]['range_resolution_m']

        self.fall_vel_threshold = 0.5
        self.move_vel_threshold = 0.2
        self.dim = 2
        self.num_rx = 3
        self.anglelen = self.numchirps * 2
        self.philen = self.numchirps * 2
        
        self.dist_points = np.linspace(0, self.max_range_m / 2, int(self.chirpsamples / 2))
        self.vel_point = np.linspace(-self.max_speed_m_s, self.max_speed_m_s, self.numchirps * 2)
        self.theta_vec = np.linspace(-np.pi / 2, np.pi / 2, self.anglelen)
        self.y_spec = np.zeros(int(len(self.theta_vec)), dtype=complex)
        self.a_thetta = np.exp(-1j * np.pi * np.sin(self.theta_vec))

        self.vel_itemindex_move = np.where(np.abs(np.array(self.vel_point)) > self.move_vel_threshold)
        self.vel_itemindex_fall = np.where(np.abs(np.array(self.vel_point)) > self.fall_vel_threshold)
        vel_threshold_ml = 2
        self.vel_itemindex_ml = np.where(np.abs(np.array(self.vel_point)) < vel_threshold_ml)

        self.count_vel = 0
        self.final_count = 0
        self.absence_count = 0
        self.Present_count = 0
        self.abnorm_count = 0
        self.det_count = 0
        self.bed_points = []
        self.bed_cluster = []

    @profile(stream=open('mp_run_bm2.log', 'w+'))
    def run_benchmark2(self, queue, total_number_of_files):
        # if psutil.LINUX:
        #     obj = Worker(logFilePath=str(processingPath) + "/logs/")

        STEP_TIME = 4
        det_count_thereshold = 10
        COUNT_VEL = 1
        COUNT_MOVE = 1
        det_count_move = 10
        processed_files = 0

        final_decision_for_fall_thread = threading.Thread(
            name="Final_Fall_Detection",
            target=self.final_decision_for_fall,
            args=(datetime.now(),)
        )
        final_decision_for_fall_thread.start()

        first_run = True
        i = 0
        j = 0
        mm = 0
        bb = 0
        rs = 0
        self.bbs = 0
        self.gbc = 0
        self.rdsk = 0
        count_step_time = 0
        ml_rdm_queue_list = []
        fall_rdm_queue_list = []
        move_rdm_queue_list = []
        ml_rdm_queue_date = []
        fall_rdm_queue_date = []
        move_rdm_queue_date = []
        start_time_stamp = datetime.now()

        while not self.unzip_event.is_set() or not self.queue.empty():
            if not self.queue.empty():
                data = self.queue.get()
                self.processed_files += 1
                percentage = (self.processed_files / total_number_of_files) * 100
                print(f"({percentage:.2f}%) Processed {self.processed_files}/{total_number_of_files} files")

                if first_run:
                    first_run = False
                    self.initialize_parameters(data)
                    RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
                    RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
                    vel_old = np.zeros((self.numchirps * 2, 1), dtype=complex)
                    Range_Dopp_prof = []
                    bed_cluster_in = np.load(os.getcwd() + "/src/processing/bed_clusterRIA.npy")
                    range_window = blackmanharris(self.chirpsamples).reshape(1, self.chirpsamples)
                    doppler_window = blackmanharris(self.numchirps).reshape(1, self.numchirps)
                    dopp_avg = np.zeros((self.chirpsamples // 2, self.numchirps * 2, self.num_rx), dtype=complex)
                    moving_avg_alpha = 0.6
                    mti_alpha = 1.0

                fft_2 = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
                for iAnt in range(3):
                    ele = "RX" + str(iAnt + 1)
                    mat = data[ele]
                    avgs = np.average(mat, 1).reshape(self.numchirps, 1)
                    mat = mat - avgs
                    mat = np.multiply(mat, range_window)
                    zp1 = np.pad(mat, ((0, 0), (0, self.chirpsamples)), 'constant')
                    range_fft = np.fft.fft(zp1) / self.chirpsamples
                    range_fft = 2 * range_fft[:, range(int(self.chirpsamples / 2))]
                    fft1d = np.transpose(range_fft)
                    fft1d = np.multiply(fft1d, doppler_window)
                    zp2 = np.pad(fft1d, ((0, 0), (0, self.numchirps)), 'constant')
                    fft2d = np.fft.fft(zp2) / self.numchirps
                    dopp_avg[:, :, iAnt] = (fft2d * moving_avg_alpha) + (dopp_avg[:, :, iAnt] * (1 - moving_avg_alpha))
                    fft2d_mti = fft2d - (dopp_avg[:, :, iAnt] * mti_alpha)
                    dopplerfft = np.fft.fftshift(fft2d_mti, (1,))
                    Range_Dopp_prof.append(dopplerfft)
                    fft_2 = fft_2 + dopplerfft

                ml_rdm_queue_list.append(fft_2)
                ml_rdm_queue_date.append('ml_' + data["radar_config"]['date'] + ".pickle.bz2")
                
                if len(ml_rdm_queue_list) == STEP_TIME + 1:
                    self.thread_pool.submit(self.run_ml, STEP_TIME, count_step_time, list(ml_rdm_queue_list), list(ml_rdm_queue_date))
                    i += 1
                    ml_rdm_queue_list.clear()
                    ml_rdm_queue_date.clear()

                fall_rdm_queue_list.append(fft_2)
                fall_rdm_queue_date.append('fall_' + data["radar_config"]['date'] + ".pickle.bz2")

                if len(fall_rdm_queue_list) == COUNT_VEL * det_count_thereshold:
                    self.thread_pool.submit(self.run_signal_processing_fall_detection, COUNT_VEL, det_count_thereshold, list(fall_rdm_queue_list), list(fall_rdm_queue_date))
                    j += 1
                    fall_rdm_queue_list.clear()
                    fall_rdm_queue_date.clear()


                move_rdm_queue_list.append(fft_2)
                move_rdm_queue_date.append('move_' + data["radar_config"]['date'] + ".pickle.bz2")

                if len(move_rdm_queue_list) == COUNT_MOVE * det_count_move:
                    self.thread_pool.submit(self.run_signal_processing_move, COUNT_MOVE, det_count_move, list(move_rdm_queue_list), "move_" + str(mm), list(move_rdm_queue_date))
                    move_rdm_queue_list.clear()
                    move_rdm_queue_date.clear()
                    mm += 1

                self.thread_pool.submit(self.reset_thrd_val, mm, bb, i, j, rs, self.rdsk)
                rs += 1

                if self.count_vel == 10:
                    RngDopp_sum = fft_2 + RngDopp_sum
                    RngAzMat_sum = self.Capon_beaformer(Range_Dopp_prof) + RngAzMat_sum
                    del Range_Dopp_prof[:]
                    self.Position_clustering(RngAzMat_sum)

                    vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                    # print("===================================================================================")
                    # print(f"abs(vel_old):{abs(vel_old)}")
                    # print()
                    # print(f"abs(vel_sum): {abs(vel_sum)}")
                    corr, _ = pearsonr(abs(vel_old), abs(vel_sum))
                    # print(corr)

                    if corr > 0.6:
                        self.Present_count = self.Present_count + 1
                    else:
                        self.absence_count = self.absence_count + 1
                    
                    # print(f"self.Present_count: {self.Present_count}")
                    # print(f"self.absence_count: {self.absence_count}")
                    # print("===================================================================================")
                    vel_old = vel_sum 
                    self.count_vel = 0
                    self.final_count = self.final_count + 1
                    # RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
                    # RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
                    RngDopp_sum.fill(0)  # Reset for the next iteration
                    RngAzMat_sum.fill(0)
                else:
                    # self.count_vel += 1
                    # RngDopp_sum = fft_2 + RngDopp_sum
                    # RngAzMat_sum = self.Capon_beaformer(Range_Dopp_prof) + RngAzMat_sum
                    # del Range_Dopp_prof[:]
                    self.count_vel += 1
                    RngDopp_sum += fft_2
                    RngAzMat_sum += self.Capon_beaformer(Range_Dopp_prof)

                if self.final_count == 10:
                    self.logging_info(f"CPU utilization: {psutil.cpu_percent(percpu=True)}%")
                    self.Time_bed_previous = self.Time_bed
                    self.Time_bed = str(data["radar_config"]['date'])
                    T_detct = self.Present_count + self.absence_count
                    thr = self.Present_count / T_detct
                    
                    info='Duration= '+  str(datetime.now() - start_time_stamp)
                    threading.Thread(
                            name="Duration" + '_' + str(self.Time_bed),
                            target=self.logging_info,
                            args=(info,),
                        ).start()
                    
                    if thr > 0.7:
                        self.PAFlag = True
                        info=str(self.Time_bed) + '  PAD >>>>>> ' + " Present" + ': '
                        
                        threading.Thread(
                            name="Present" + '_' + str(self.Time_bed),
                            target=self.logging_info,
                            args=(info,),
                        ).start()
                        threading.Thread(
                            name="bed_" + '_' + str(bb),
                            target=self.Bed_Detection,
                            args=(bed_cluster_in, self.Time_bed),
                        ).start()
                        bb += 1
                    else:
                        self.In_bed = False
                        info=str(self.Time_bed) + '  PAD >>>>>> ' + " Absent" + ': '
                        threading.Thread(
                            name="Absent" + '_' + str(self.Time_bed),
                            target=self.logging_info,
                            args=(info,),
                        ).start()
                    self.final_count = 0
                    self.Present_count = 0
                    self.absence_count = 0
                    T_detct = 0
                    start_time_stamp = datetime.now()

                
                # Memory management
                gc.collect()  # Explicitly collect garbage
                self.queue.task_done()
                if self.processed_files >= total_number_of_files:
                    print("ALL FILES ARE PROCESSED. EXIT LOOP..")
                    break
            else:
                while self.queue.empty() and not self.unzip_event.is_set():
                    threading.Event().wait(0.01)

        # obj.sendFinish()
        self.queue.join()
        print("Benchmarking completed. All files processed.")
        print("GC STATUS AFTER FINISH")
        print(gc.get_stats())
        return
    

    @profile(stream=open('mp_run_ml.log', 'w+'))
    def run_ml(self, step_time, count_step_time, ml_rdm_queue_list, ml_rdm_queue_date):
        step_time = int(step_time)
        output = 'detection'
        RDM_vector = []
        dates = []

        for i in range(0, len(ml_rdm_queue_list)):
            tmp_rdm = ml_rdm_queue_list[i]
            dates.append(ml_rdm_queue_date[i].split('ml_')[1].split('.pickle.bz2')[0])
            New_rng = (tmp_rdm[:, self.vel_itemindex_ml])
            RDM = abs(New_rng.reshape(New_rng.shape[0], New_rng.shape[2])) + 0
            if count_step_time == step_time:
                # output = self.Model_Test(RDM_vector)
                output = self.model_test(RDM_vector)
                count_step_time = 0
                del RDM_vector[:]
            else:
                count_step_time += 1
                vector_tmp = RDM.reshape(RDM.shape[0] * RDM.shape[1])
                RDM_vector.append((vector_tmp))

        if output == 'Moving':
            logging.info(dates[-1] + " >>>>>>ML " + output)

        for date in dates:
            if date:
                self.insert_into_ml_result_list({'date': date, 'output': output})

        del dates[:]

        if len(self.ml_result_list) >= 5120:
            self.remove_from_ml_result_list()
            threading.Thread(
                name="garbage" + '_' + str(self.gbc),
                target=self.gc_collect
            ).start()
            self.gbc += 1

        return

    @profile(stream=open('mp_run_fall.log', 'w+'))
    def run_signal_processing_fall_detection(self, count_vel, det_count_thereshold, fall_rdm_queue_list, fall_rdm_queue_date):
        first_run = True
        count_vel_thereshold = int(count_vel)

        for i in range(0, len(fall_rdm_queue_list)):
            if first_run:
                first_run = False
                count_vel = 0
                det_count = 0
                abnorm_count = 0
                vel_old = (np.ones((self.numchirps * 2, 1), dtype=complex)) * (np.random.rand((self.numchirps * 2)))
                vel_filt_old = vel_old[self.vel_itemindex_fall]
                vel_point_filt = self.vel_point[self.vel_itemindex_fall]
                RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)

            fft_2 = fall_rdm_queue_list[i]
            date = fall_rdm_queue_date[i].split('fall_')[1].split('.pickle.bz2')[0]

            if count_vel == count_vel_thereshold - 1:
                RngDopp_sum = fft_2 + RngDopp_sum
                vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                vel_filt = vel_sum[self.vel_itemindex_fall]
                corr_fall, _ = pearsonr(abs(vel_filt), abs(vel_filt_old))

                if corr_fall > 0.5:
                    ng = 4
                    nt = 8
                    det_vel_sum = self.SO_CFAR(vel_filt, ng, nt)
                    ind_CFAR_vel = np.argwhere(det_vel_sum == 1)
                    det_vel = (np.abs(vel_point_filt[ind_CFAR_vel]))
                    filtind_2 = (np.argwhere(det_vel < 2.8))
                    if len(filtind_2) > 0:
                        abnorm_count += 1

                vel_filt_old = vel_filt + 0
                count_vel = 0
                det_count += 1
                RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            else:
                count_vel += 1
                RngDopp_sum = fft_2 + RngDopp_sum

            if det_count == det_count_thereshold:
                if abnorm_count > 1:
                    self.suspected_fall_queue.put(str(date))
                    logging.info(date + " >>>>>> suspected fall")
                abnorm_count = 0
                det_count = 0

        return

    def remove_from_ml_result_list(self):
        self.ml_result_list = self.ml_result_list[len(self.ml_result_list) // 2:]
        gc.collect()
        return

    def gc_collect(self):
        gc.collect()
        return

    @profile(stream=open('mp_run_move.log', 'w+'))
    def run_signal_processing_move(self, count_move, det_count_thereshold, move_rdm_queue_list, function_name, move_rdm_queue_date):
        first_run = True
        det_count_thereshold = 50
        count_move_thereshold = int(count_move)
        for i in range(0, len(move_rdm_queue_list)):
            if first_run:
                first_run = False
                count_vel = 0
                vel_filt_old = self.vel_point[self.vel_itemindex_move]
                vel_point_filt = self.vel_point[self.vel_itemindex_move]

            date = move_rdm_queue_date[i].split('move_')[1].split('.pickle.bz2')[0]
            fft_2 = move_rdm_queue_list[i]
            RngDopp_sum = fft_2 + 0

            if count_vel == count_move_thereshold - 1:
                RngDopp_sum = fft_2 + RngDopp_sum
                vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                vel_filt = vel_sum[self.vel_itemindex_move]
                corr_move, _ = pearsonr(abs(vel_filt), abs(vel_filt_old))

                if corr_move > 0.5:
                    ng = 4
                    nt = 8
                    det_vel_sum = self.SO_CFAR(vel_filt, ng, nt)
                    ind_CFAR_vel = np.argwhere(det_vel_sum == 1)
                    det_vel = (np.abs(vel_point_filt[ind_CFAR_vel]))
                    filtind_2 = (np.argwhere(det_vel < 2.8))
                    if len(filtind_2) > 0:
                        self.move_Cnt += 1

                vel_filt_old = vel_filt + 0
                count_vel = 0
                self.mv_det_count += 1
                RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            else:
                count_vel += 1
                RngDopp_sum = fft_2 + RngDopp_sum

        del move_rdm_queue_list[:]
        if self.mv_det_count == det_count_thereshold:
            if self.In_bed and self.move_Cnt > 5 and self.Bed_Cnt > 3:
                logging.info(date + " >>>>>>>> Bed movement is detected")
            self.mv_det_count = 0
            self.move_Cnt = 0

        return

    @profile(stream=open('mp_final_fall.log', 'w+'))
    def final_decision_for_fall(self, start_date):
        while True:
            if not self.suspected_fall_queue.empty():
                date = self.suspected_fall_queue.get()
                final_decision = False

                while not final_decision:
                    index_bed_sfd = self.find_bed_result_list_index(date)
                    index = self.find_ml_result_list_index(date)
                    if index is not None:
                        try:
                            if self.ml_result_list[index + 250]['date']:
                                walking_counter = 0
                                for i in range(index, index + 250):
                                    output = self.extract_output_from_ml_result_list(self.ml_result_list[i]['date'])
                                    if str.lower(output) == 'moving':
                                        walking_counter += 1
                                if walking_counter <= 20:
                                    if index - 40 >= 0:
                                        walking_counter = 0
                                        for i in range(index - 40, index + 1):
                                            output = self.extract_output_from_ml_result_list(self.ml_result_list[i]['date'])
                                            if str.lower(output) == 'moving':
                                                walking_counter += 1
                                                if walking_counter == 1 and not self.In_bed:
                                                    if index_bed_sfd is not None:
                                                        final_decision = True
                                                        logging.info(date + " >>>>>>  Suspected Fall Approved")
                                                        break
                                                    else:
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
            else:
                time.sleep(0.1)
        return

    def find_bed_result_list_index(self, date):
        index = 0
        for i in range(len(self.bed_result_list)):
            if self.bed_result_list[i]['date'] >= date:
                index = i
                return index
        return None

    def find_ml_result_list_index(self, date):
        index = 0
        for i in range(len(self.ml_result_list)):
            if self.ml_result_list[i]['date'] == date:
                index = i
                return index
        return None

    def extract_output_from_ml_result_list(self, date):
        for i in range(len(self.ml_result_list)):
            if self.ml_result_list[i]['date'] == date:
                return self.ml_result_list[i]['output']
        return

    # def Model_Test(self, x_test):
    #     output = ''
    #     X_test = ((np.array(x_test)))
    #     X2 = X_test.reshape(1, X_test.shape[0] * X_test.shape[1])
    #     try:
    #         X2 = scaler.transform(X2)
    #     except:
    #         return output
    #     X_test = X2.reshape(1, X_test.shape[0], X_test.shape[1])

    #     old_stdout = sys.stdout
    #     sys.stdout = open(os.devnull, "w")

    #     pred_result = model(X_test, training=False)

    #     sys.stdout = old_stdout

    #     y_pred = np.argmax(pred_result, axis=-1)
    #     tf.keras.backend.clear_session()

    #     if y_pred == 0:
    #         output = 'Empty'
    #     elif y_pred == 1:
    #         output = 'Moving'
    #     elif y_pred == 2:
    #         output = 'Stationary'
    #     elif y_pred == 3:
    #         output = 'In_bed'
    #     elif y_pred == 4:
    #         output = 'on_floor'
    #     else:
    #         output = 'unknown'

    #     del X_test
    #     del X2
    #     del y_pred

    #     return output

    def model_test(self, X_test):
        output = ''
        X_test = ((np.array(X_test)))
        X2 = X_test.reshape(1, -1)
        expected_features = scaler.n_features_in_

        if X2.shape[1] != expected_features:
            if X2.shape[1] % expected_features == 0:
                num_splits = X2.shape[1] // expected_features
                X2 = X2.reshape(num_splits, expected_features)
            else:
                print("[TEMPORARY] SOMETHING WRONG WITH RESHAPE")
                return -1

        X2 = scaler.transform(X2)
        X_test = X2.reshape(-1, 4, 5440).astype(np.float32)

        pred_result = np.array(ort_session.run(None, {"lstm_input:0": X_test})).flatten()
        y_pred = np.argmax(pred_result, axis=-1)

        if y_pred == 0:
            output = 'Empty'
        elif y_pred == 1:
            output = 'Moving'
        elif y_pred == 2:
            output = 'Stationary'
        elif y_pred == 3:
            output = 'In_bed'
        elif y_pred == 4:
            output = 'on_floor'
        else:
            output = 'unknown'

        del X_test, X2, pred_result, y_pred
        gc.collect()

        return output
    
    def reset_thrd_val(self, mm, bb, i, j, rs, rngazm):
        if mm > 2000:
            mm = 0
        if bb > 2000:
            bb = 0
        if i > 2000:
            i = 0
        if j > 2000:
            j = 0
        if rs > 2000:
            rs = 0
        if rngazm > 2000:
            rngazm = 0
        if self.gbc > 2000:
            self.gbc = 0
        if self.rdsk > 2000:
            self.rdsk = 0

    def Capon_beaformer(self, Range_Dopp_prof):
        RngAzMat = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
        Rang_Prof_Az = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec)), int(self.dim)), dtype=complex)

        Rang_Prof_Az[:, :, 0] = Range_Dopp_prof[0]
        Rang_Prof_Az[:, :, 1] = Range_Dopp_prof[2]

        RangeMatrix_her = 1 / self.numchirps * np.conjugate(Rang_Prof_Az.transpose([0, 2, 1]))

        for rr in range(0, int(self.chirpsamples / 2)):
            inv_R_hat = np.linalg.inv(RangeMatrix_her[rr, ...] @ Rang_Prof_Az[rr, ...])

            for jj in range(len(self.theta_vec)):
                a_hat = np.array([1, self.a_thetta[jj]])
                self.y_spec[jj] = 1 / (a_hat.conjugate().transpose() @ inv_R_hat @ a_hat)

            RngAzMat[rr, :] = self.y_spec

        return RngAzMat

    def SO_CFAR(self, input_vector, ng, nt):
        P_fa = 0.3
        input_width = input_vector.size

        if input_width < 2 * ng + 2 * nt + 1:
            raise Exception('The length of the input should be > 2*ng + 2*nt + 1')

        above_threshold = np.zeros(input_width)
        above_threshold[:ng + 1] = np.nan
        above_threshold[-ng - 1:] = np.nan

        for testCell in range(ng + 1, input_width - ng - 1):
            if testCell - ng - nt >= 0 and testCell + ng + nt < input_width:
                lower_window = input_vector[testCell - ng - nt:testCell - ng]
                upper_window = input_vector[testCell + ng:testCell + ng + nt]
            else:
                if testCell - ng - nt < 0:
                    lower_window = input_vector[0:testCell - ng]
                    upper_window = input_vector[testCell + ng:testCell + ng + nt]
                elif testCell + ng + nt > input_width:
                    lower_window = input_vector[testCell - ng - nt:testCell - ng]
                    upper_window = input_vector[testCell + ng:input_width]

            n1 = lower_window.size
            n2 = upper_window.size
            n_thr = n1 + n2
            lower_sum = np.sum(lower_window)
            upper_sum = np.sum(upper_window)
            summation = np.divide((lower_sum + upper_sum), n_thr)
            alpha = n_thr * (((P_fa ** (np.divide(-1, n_thr)))) - 1)

            if input_vector[testCell] > np.multiply(summation, alpha):
                above_threshold[testCell] = 1

            del summation

        return above_threshold

    def logging_info(self, info):
        logging.info(info)
        return

    def Position_clustering(self, RngAzMat_sum):
        thresholded_result = self.cfar_2d(np.abs(RngAzMat_sum), guard_cells=2, training_cells=8, threshold_factor=1.5)
        det_index = np.array(np.where(thresholded_result == 1))

        if det_index.shape[1] > 0:
            dbscan = DBSCAN(eps=1, min_samples=3, metric='euclidean')
            X_dbscan = np.transpose(det_index)
            _ = dbscan.fit(X_dbscan)
            labels = dbscan.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)

            if n_clusters_ > 1:
                logging.info('  Counting >>>>>> ' + " multiple people")

            if n_clusters_ == 1:
                for k in unique_labels:
                    class_member_mask = labels == k
                    xy = X_dbscan[class_member_mask]
                    self.bed_points.append(xy)

        return self.bed_points

    @profile(stream=open('mp_bed_detection.log', 'w+'))
    def Bed_Detection(self, bed_cluster_in, Time_bed):
        self.In_bed_previous = self.In_bed
        all_data_points = [point for cluster in self.bed_points for point in cluster]
        point_counts = Counter(map(tuple, all_data_points))
        most_repeated_points = point_counts.most_common()

        if not most_repeated_points:
            logging.info(self.Time_bed + '  Status >>>>>> ' + " Not_detected")
            self.In_bed = False
            threading.Thread(
                name="bed_status_" + '_' + str(self.bbs),
                target=self.check_in_bed_transition,
                args=(),
            ).start()
            self.bbs += 1

        else:
            most_repeated_cluster = most_repeated_points[0][0]
            most_repeated_cluster_set = set(most_repeated_cluster)
            most_repeated_cluster_points = [point for cluster in self.bed_points for point in cluster if set(point) == most_repeated_cluster_set]

            x_coords, y_coords = zip(*most_repeated_cluster_points)
            XYbed = [x_coords, y_coords]
            unique_bed_cluster_tuples = [data_point if isinstance(data_point, tuple) else (data_point,) for data_point in XYbed]

            unique_bed_cluster_array = np.array(unique_bed_cluster_tuples)
            dist_to_bed = self.distance_to_cluster(np.unique(unique_bed_cluster_array), bed_cluster_in)

            if dist_to_bed < 2:
                self.In_bed = True
                self.Bed_Cnt += 1
                logging.info(self.Time_bed + '  Status >>>>>> ' + " In_bed")
            else:
                self.In_bed = False
                threading.Thread(
                    name="bed_status_" + '_' + str(self.bbs),
                    target=self.check_in_bed_transition,
                    args=(),
                ).start()
                self.bbs += 1

            self.insert_into_bed_result_list({'date': self.Time_bed, 'output': self.In_bed})

            if len(self.bed_result_list) >= 20:
                self.remove_from_bed_result_list()
            del self.bed_points[:]

        return

    def check_in_bed_transition(self):
        if psutil.LINUX:
            pass

        if self.In_bed_previous and not self.In_bed:
            self.check_Pre_movement()
            if self.Pre_Movment_detected:
                self.In_bed = False
                self.Bed_Cnt = 0
                self.insert_into_bed_result_list({'date': self.Time_bed, 'output': self.In_bed})
            else:
                self.In_bed = True
                self.Bed_Cnt += 1
                self.insert_into_bed_result_list({'date': self.Time_bed, 'output': self.In_bed})
                logging.info("301 : "+self.Time_bed + ' -> ' + " In_bed")

        if psutil.LINUX:
            pass

        return

    def insert_into_bed_result_list(self, dict):
        index = len(self.bed_result_list)
        for i, entry in enumerate(self.bed_result_list):
            if entry['date'] > dict['date']:
                index = i
                break
        self.bed_result_list.insert(index, (dict))
        return self.bed_result_list

    def remove_from_bed_result_list(self):
        self.bed_result_list = self.bed_result_list[len(self.bed_result_list) // 2:]
        gc.collect()
        return

    def check_Pre_movement(self):
        index1 = self.find_ml_result_list_index2(self.Time_bed_previous)
        threshold_bed_transition = len(self.ml_result_list) - index1 - 1

        moving_counter = 0
        for i in range(index1, index1 + threshold_bed_transition):
            output = self.extract_output_from_ml_result_list(self.ml_result_list[i]['date'])
            if (str.lower(output) == 'moving'):
                moving_counter += 1

        if moving_counter > 6:
            self.Pre_Movment_detected = True
        else:
            self.Pre_Movment_detected = False

        return

    def distance_to_cluster(self, point, cluster):
        distances = [self.euclidean_distance(point, cluster_point) for cluster_point in cluster]
        return min(distances)

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def find_ml_result_list_index2(self, date):
        index = 0
        for i in range(len(self.ml_result_list)):
            if self.ml_result_list[i]['date'] >= date:
                index = i
                return index
        return None

    def SO_CFAR(self, input_vector, ng, nt):
        P_fa = 0.3
        input_width = input_vector.size

        if input_width < 2 * ng + 2 * nt + 1:
            raise Exception('The length of the input should be > 2*ng + 2*nt + 1')

        above_threshold = np.zeros(input_width)
        above_threshold[:ng + 1] = np.nan
        above_threshold[-ng - 1:] = np.nan

        for testCell in range(ng + 1, input_width - ng - 1):
            if testCell - ng - nt >= 0 and testCell + ng + nt < input_width:
                lower_window = input_vector[testCell - ng - nt:testCell - ng]
                upper_window = input_vector[testCell + ng:testCell + ng + nt]
            else:
                if testCell - ng - nt < 0:
                    lower_window = input_vector[0:testCell - ng]
                    upper_window = input_vector[testCell + ng:testCell + ng + nt]
                elif testCell + ng + nt > input_width:
                    lower_window = input_vector[testCell - ng - nt:testCell - ng]
                    upper_window = input_vector[testCell + ng:input_width]

            n1 = lower_window.size
            n2 = upper_window.size
            n_thr = n1 + n2
            lower_sum = np.sum(lower_window)
            upper_sum = np.sum(upper_window)
            summation = np.divide((lower_sum + upper_sum), n_thr)
            alpha = n_thr * (((P_fa ** (np.divide(-1, n_thr)))) - 1)

            if input_vector[testCell] > np.multiply(summation, alpha):
                above_threshold[testCell] = 1

            del summation

        return above_threshold

    def Position_clustering(self, RngAzMat_sum):
        thresholded_result = self.cfar_2d(np.abs(RngAzMat_sum), guard_cells=2, training_cells=8, threshold_factor=1.5)
        det_index = np.array(np.where(thresholded_result == 1))

        if det_index.shape[1] > 0:
            dbscan = DBSCAN(eps=1, min_samples=3, metric='euclidean')
            X_dbscan = np.transpose(det_index)
            _ = dbscan.fit(X_dbscan)
            labels = dbscan.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)

            if n_clusters_ > 1:
                logging.info('  Counting >>>>>> ' + " multiple people")

            if n_clusters_ == 1:
                for k in unique_labels:
                    class_member_mask = labels == k
                    xy = X_dbscan[class_member_mask]
                    self.bed_points.append(xy)

        return self.bed_points

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
