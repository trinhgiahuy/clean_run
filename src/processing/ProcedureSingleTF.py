import bz2
import pickle
import glob
import queue
import numpy as np
import os
import os.path
import psutil
import logging
import tensorflow as tf
from sklearn.cluster import DBSCAN
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
import onnx
import onnxruntime as ort


# Huy's monitoring framework
from src.processing.tools.my_track_stats import Worker

import time
result_log_path = 'VAL_ONNX_FULL_NO_THREAD_RUN_TIME.log'
if psutil.LINUX:
    processingPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/SE455-Lenovo-Server-Code/src/processing/tools"

# print(f"processingPath: {processingPath}")

def load_model_and_scaler():
    base_dir = os.path.join(os.getcwd(), "src", "processing", "ML", "models", "LSTM_4class")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    model_path = os.path.join(base_dir, "model_LSTM_4class.h5")

    with open(scaler_path, 'rb') as file:
        scaler = load(file)
    model = tf.keras.models.load_model(model_path)

    return scaler, model

scaler, model = load_model_and_scaler()
# start_time_stamp = datetime.now()

class ProcedureSingleTF:

    def __init__(self, dataPath: str):
        self.start_time_stamp = datetime.now()
        # self.queue = queue.Queue()
        self.suspected_fall_queue = queue.Queue(maxsize=2048)
        self.ml_result_list = []
        self.bed_result_list = []
        self.In_bed = False

        self.dataPath = dataPath
        self.First_Run = False
        self.pickleList = glob.glob(self.dataPath)
        self.files = []
        self.processed_files = 0
        self.number_of_files = len(self.pickleList)

        self.process_file()
        self.end_time_stamp = datetime.now()
        total_duration = self.end_time_stamp - self.start_time_stamp
        print(f"Benchmarking completed. All files processed in {total_duration}.")
        with open(result_log_path, 'a') as log_file:
            log_file.write(f"\nTotal processing time: {total_duration}\n")
    # def unzip_files(self):
    #     for file in self.pickleList:
    #         self.process_file(file)

    def process_file(self):

        first_run = True
        for file in self.pickleList:
            # queue = queue.Queue()
            # with bz2.BZ2File(fileName, 'rb') as ifile:
            if first_run:
                first_run_flag = True
                first_run = False
            else:
                first_run_flag = False

            # with open(file, "rb") as ifile:
            with bz2.BZ2File(file, 'rb') as ifile:

                pickle_data = pickle.load(ifile)
                # queue.put(pickle_data)
                # logging.info(f"Added file {fileName} to queue. Current queue size: {self.queue.qsize()}")

                # Call function run_benchmark2 to process queue
                # print(f"Call function run_benchmark2 to process data with first_run_flag: {first_run_flag}")
                self.run_benchmark2(pickle_data=pickle_data,first_run=first_run_flag)


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

        self.vel_filt_old = self.vel_point[self.vel_itemindex_move]
        self.vel_point_filt = self.vel_point[self.vel_itemindex_move]

        self.vel_old = np.zeros((self.numchirps * 2, 1), dtype=complex)
        self.RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
        self.RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
        self.bed_cluster_in = np.load(os.getcwd() + "/src/processing/bed_clusterRIA.npy")

    @profile(stream=open('VAL_mp_run_bm2.log', 'w+'))
    def run_benchmark2(self, pickle_data, first_run):
        # if psutil.LINUX:
        #     obj = Worker(logFilePath=str(processingPath) + "/logs/")

        STEP_TIME = 4
        det_count_thereshold = 10
        COUNT_VEL = 1
        COUNT_MOVE = 1
        det_count_move = 10
        processed_files = 0

        # first_run = True


        # if not queue.empty():
        # data = queue.get()
        data = pickle_data
        self.processed_files += 1
        percentage = (self.processed_files / self.number_of_files) * 100
        print(f"({percentage:.2f}%) Processed {self.processed_files}/{self.number_of_files} files")

        if first_run:
            first_run = False
            self.initialize_parameters(data)
            self.bbs = 0
            self.gbc = 0
            self.rdsk = 0
            self.count_step_time = 0
            self.ml_rdm_queue_list = []
            self.fall_rdm_queue_list = []
            self.move_rdm_queue_list = []
            self.ml_rdm_queue_date = []
            self.fall_rdm_queue_date = []
            self.move_rdm_queue_date = []
            self.Range_Dopp_prof = []
            self.i = 0
            self.j = 0
            self.mm = 0
            self.bb = 0
            self.rs = 0

            
        # start_time_stamp = datetime.now()
        # ml_rdm_queue_list=self.ml_rdm_queue_list.copy()
        # ml_rdm_queue_date=self.ml_rdm_queue_date.copy()
        # print(f"len:{len(ml_rdm_queue_list)}")
        # time.sleep(1)



        # CAREFUL WITH THESE VARIABLE AS IT HAS TO BE UPDATED OVER ITERATIONS
        # RECHECK ALL THESE BELOW WHICH MAY CAUSE BUGS
        # CONSIDER SWITCHING ALL INTO SELF.
        
        # RngDopp_sum = self.RngDopp_sum
        # RngAzMat_sum = self.RngAzMat_sum
        # vel_old = self.vel_old
        #Range_Dopp_prof = self.Range_Dopp_prof
        
 

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
            self.Range_Dopp_prof.append(dopplerfft)
            fft_2 = fft_2 + dopplerfft


        # --------------------------------run_ml------------------------------------------------------------------
        self.ml_rdm_queue_list.append(fft_2)
        self.ml_rdm_queue_date.append('ml_' + data["radar_config"]['date'] + ".pickle.bz2")
        # print(f"__________{len(ml_rdm_queue_list)}")

        # [CALL AFTER 5 FRAMES]
        if len(self.ml_rdm_queue_list) == STEP_TIME + 1:
            self.run_ml(STEP_TIME, self.count_step_time, self.ml_rdm_queue_list.copy(), self.ml_rdm_queue_date.copy())
            self.i += 1
            self.ml_rdm_queue_date.clear()
            self.ml_rdm_queue_list.clear()


        # --------------------------------run_signal_processing_fall_detection-----------------------------------
        self.fall_rdm_queue_list.append(fft_2)
        self.fall_rdm_queue_date.append('fall_' + data["radar_config"]['date'] + ".pickle.bz2")
        # print(f"**fall_rdm_queue_list[{len(self.fall_rdm_queue_list)}] {COUNT_VEL * det_count_thereshold}")

        # [CALL AFTER 10 FRAMES]
        if len(self.fall_rdm_queue_list) == COUNT_VEL * det_count_thereshold:
            # print(f"CALL run_signal_processing_fall_detection with first_run {first_run}")
            first_run=True
            self.run_signal_processing_fall_detection(COUNT_VEL, det_count_thereshold, self.fall_rdm_queue_list.copy(), self.fall_rdm_queue_date.copy() 
                                                      ,first_run, 
                                                      self.RngDopp_sum,
                                                      vel_filt_old=self.vel_filt_old,
                                                      vel_point_filt=self.vel_point_filt
                                                      )
            self.j += 1
            self.fall_rdm_queue_date.clear()
            self.fall_rdm_queue_list.clear()
            

        # --------------------------------run_signal_processing_fall_detection-----------------------------------
        self.move_rdm_queue_list.append(fft_2)
        self.move_rdm_queue_date.append('move_' + data["radar_config"]['date'] + ".pickle.bz2")
        # print(f"--move_rdm_queue_list[{len(self.move_rdm_queue_list)}] {COUNT_MOVE * det_count_move}")

        # [CALL AFTER 10 FRAMES]
        if len(self.move_rdm_queue_list) == COUNT_MOVE * det_count_move:
            # print("CALL run_signal_processing_move")
            # First_run always true here, change later inside the function
            first_run=True
            self.run_signal_processing_move(COUNT_MOVE, det_count_move, self.move_rdm_queue_list.copy(), "move_" + '_' + str(self.mm), self.move_rdm_queue_date.copy(), 
                                            first_run) 
                                            # self.RngDopp_sum,
                                            # vel_filt_old=self.vel_filt_old,
                                            # vel_point_filt=self.vel_point_filt)
            self.mm += 1
            self.move_rdm_queue_date.clear()
            self.move_rdm_queue_list.clear()
            

        self.reset_thrd_val(self.mm, self.bb, self.i, self.j, self.rs, self.rdsk)
        self.rs += 1


        # ===================================================================================
        if self.count_vel == 10:
            self.RngDopp_sum = fft_2 + self.RngDopp_sum
            # RngDopp_sum = self.RngDopp_sum
            self.RngAzMat_sum = self.Capon_beaformer(self.Range_Dopp_prof) + self.RngAzMat_sum
            self.Range_Dopp_prof.clear()
            self.Position_clustering(self.RngAzMat_sum)

            self.vel_sum = ((np.abs(self.RngDopp_sum)).sum(axis=0)) + 0
            # vel_sum = self.vel_sum
            # print(f"abs(vel_old):{abs(vel_old)}")
            # print()
            # print(f"abs(vel_sum): {abs(self.vel_sum)}")
            corr, _ = pearsonr(abs(self.vel_old), abs(self.vel_sum))
            # print(corr)
            
            if corr > 0.6:
                self.Present_count += 1
            else:
                self.absence_count += 1

            print(f"self.Present_count: {self.Present_count}")
            print(f"self.absence_count: {self.absence_count}")
            print("===================================================================================")
            self.vel_old = self.vel_sum + 0
            self.count_vel = 0
            self.final_count += 1

            self.RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            self.RngAzMat_sum = np.zeros((int(self.chirpsamples / 2), int(len(self.theta_vec))), dtype=complex)
        else:
            print(f"*******Increase self.count_vel {self.count_vel} to 1")
            self.count_vel += 1
            self.RngDopp_sum = fft_2 + self.RngDopp_sum
            self.RngAzMat_sum = self.Capon_beaformer(self.Range_Dopp_prof) + self.RngAzMat_sum
            self.Range_Dopp_prof.clear()
        

        print(f"__________self.final_count: {self.final_count}")
        if self.final_count == 10:
            # self.logging_info(f"CPU utilization: {psutil.cpu_percent(percpu=True)}%")
            self.Time_bed_previous = self.Time_bed
            self.Time_bed = str(data["radar_config"]['date'])
            T_detct = self.Present_count + self.absence_count
            thr = self.Present_count / T_detct

            # info = 'Duration= ' + str(datetime.now() - start_time_stamp)
            # self.logging_info(info)
            print(f"-----------------thr: {thr}")
            if thr > 0.7:
                self.PAFlag = True
                info = str(self.Time_bed) + '  PAD >>>>>> ' + " Present" + ': '
                self.logging_info(info)
                self.Bed_Detection(self.bed_cluster_in, self.Time_bed)
                self.bb += 1
            else:
                self.In_bed = False
                info = str(self.Time_bed) + '  PAD >>>>>> ' + " Absent" + ': '
                self.logging_info(info)

            self.final_count = 0
            self.Present_count = 0
            self.absence_count = 0
            # start_time_stamp = datetime.now()

        # self.queue.task_done()
        # if self.processed_files >= self.number_of_files:
            # print("ALL FILES ARE PROCESSED. EXIT LOOP..")
            # break
        # else:
        gc.collect()

        # obj.sendFinish()
        # self.queue.join()
        # print("Benchmarking completed")
        # print("GC STATUS AFTER FINISH")
        # print(gc.get_stats())
        return

    @profile(stream=open('VAL_mp_run_ml.log', 'w+'))
    def run_ml(self, step_time, count_step_time, ml_rdm_queue_list, ml_rdm_queue_date):
        # print("[C] run_ml")
        # print(f"step_time :{step_time} - count_step_time: {count_step_time} and {self.count_step_time} -  ml_rdm_queue_list: {len(ml_rdm_queue_list)} - {len(ml_rdm_queue_date)}")
        step_time = int(step_time)
        output = 'detection'
        RDM_vector = []
        dates = []

        for i in range(0, len(ml_rdm_queue_list)):
            tmp_rdm = ml_rdm_queue_list[i]
            dates.append(ml_rdm_queue_date[i].split('ml_')[1].split('.pickle.bz2')[0])
            New_rng = (tmp_rdm[:, self.vel_itemindex_ml])
            RDM = abs(New_rng.reshape(New_rng.shape[0], New_rng.shape[2])) + 0
            # print(count_step_time == step_time)
            # time.sleep(1)
            if count_step_time == step_time:
                #output = self.Model_Test(RDM_vector)
                output = self.model_test(RDM_vector)
                self.count_step_time = 0
                del RDM_vector[:]
            else:
                count_step_time += 1
                # print(f'[AFTER FALSE] count_step_time: {count_step_time} and {self.count_step_time} ')
                # time.sleep(1)
                vector_tmp = RDM.reshape(RDM.shape[0] * RDM.shape[1])
                RDM_vector.append((vector_tmp))

        # if output == 'Moving':
        #     logging.info(dates[-1] + " >>>>>>ML " + output)

        for date in dates:
            if date:
                self.insert_into_ml_result_list({'date': date, 'output': output})

        del dates[:]

        if len(self.ml_result_list) >= 5120:
            self.remove_from_ml_result_list()
            self.gc_collect()
            self.gbc += 1

        return

    @profile(stream=open('VAL_mp_run_fall.log', 'w+'))
    def run_signal_processing_fall_detection(self, count_vel, det_count_thereshold, fall_rdm_queue_list, fall_rdm_queue_date, 
                                first_run, 
                                RngDopp_sum,
                                vel_filt_old,
                                vel_point_filt):
        
        # print("=====================================================================")
        # count_vel : 1
        # det_count_thereshold: 10


        # first_run = True
        print("=====================================================================> run_signal_processing_fall_detection")
        # print(f"count_vel: {count_vel}")
        # print(f"det_count_thereshold: {det_count_thereshold}")  
        # print(f"first_run: {first_run}")
        count_vel_thereshold = int(count_vel)

        for i in range(0, len(fall_rdm_queue_list)):
            if first_run:
                first_run = False
                count_vel = 0
                det_count = 0
                abnorm_count = 0
                vel_old = (np.ones((self.numchirps * 2, 1), dtype=complex)) * (np.random.rand((self.numchirps * 2)))
                vel_filt_old = vel_old[self.vel_itemindex_fall]
                RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)

                # self.vel_old = (np.ones((self.numchirps * 2, 1), dtype=complex)) * (np.random.rand((self.numchirps * 2)))
                # self.vel_filt_old = vel_old[self.vel_itemindex_fall]
                # vel_point_filt = self.vel_point[self.vel_itemindex_fall]
                # RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)

            fft_2 = fall_rdm_queue_list[i]
            self.rdsk += 1
            date = fall_rdm_queue_date[i].split('fall_')[1].split('.pickle.bz2')[0]

            # print(f"count_vel: {count_vel} and count_vel_thereshold: {count_vel_thereshold}")
            if count_vel == count_vel_thereshold - 1:
                # [FLOW] Run this 10 times and thats it
                # print("2")
                RngDopp_sum = fft_2 + RngDopp_sum
                self.vel_sum = ((np.abs(RngDopp_sum)).sum(axis=0)) + 0
                vel_filt = self.vel_sum[self.vel_itemindex_fall]
                # print(np.shape(vel_filt))
                # print(np.shape(vel_filt_old))

                corr_fall, _ = pearsonr(abs(vel_filt), abs(vel_filt_old))
                # print(f"corr_fall: {corr_fall}")
                if corr_fall > 0.5:
                    # [FLOW] Barely go into this, which does not increase abnorm_count , which does not trigger final_decision_for_fall 
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

                # Does it mean delete the variable?
                # TODO: Test if it fine to delete RngDopp_sum locallly: del RngDopp_sum
                RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            else:
                # [FLOW] Run this once
                count_vel += 1
                RngDopp_sum = fft_2 + RngDopp_sum

            # print(f"det_count: {det_count}")

            if det_count == det_count_thereshold:
                if abnorm_count > 1:
                    # self.suspected_fall_queue.put(str(date))
                    # logging.info(date + " >>>>>> suspected fall")

                    ## HERE CALL final_decision_for_fall
                    self.final_decision_for_fall(date=date)
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

    @profile(stream=open('VAL_mp_run_move.log', 'w+'))
    def run_signal_processing_move(self, count_move, det_count_thereshold, move_rdm_queue_list, function_name, move_rdm_queue_date,
                                   first_run):
        # first_run = True
        print("=====================================================================> run_signal_processing_move ")
        det_count_thereshold = 50
        count_move_thereshold = int(count_move)
        for i in range(0, len(move_rdm_queue_list)):
            if first_run:
                # first_run = False
                count_vel = 0
                vel_filt_old = self.vel_point[self.vel_itemindex_move]
                vel_point_filt = self.vel_point[self.vel_itemindex_move]  # could be global


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
                self.RngDopp_sum = np.zeros((int(self.chirpsamples / 2), self.numchirps * 2), dtype=complex)
            else:
                count_vel += 1
                self.RngDopp_sum = fft_2 + self.RngDopp_sum

        del move_rdm_queue_list[:]
        # print(f"self.mv_det_count: { self.mv_det_count} det_count_thereshold: {det_count_thereshold}")
        if self.mv_det_count == det_count_thereshold:
            if self.In_bed and self.move_Cnt > 5 and self.Bed_Cnt > 3:
                # logging.info(date + " >>>>>>>> Bed movement is detected")
                print(date + " >>>>>>>> Bed movement is detected")
            self.mv_det_count = 0
            self.move_Cnt = 0

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

    def model_test(self, X_test):
        output = ''
        X_test = np.array(X_test)
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
        pred_result = model(X_test).numpy().flatten()
        #pred_result = np.array(ort_session.run(None, {"lstm_input:0": X_test})).flatten()
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

    @profile(stream=open('VAL_mp_bed_detection.log', 'w+'))
    def Bed_Detection(self, bed_cluster_in, Time_bed):
        print("=====================================================================> Bed_Detection")
        self.In_bed_previous = self.In_bed
        all_data_points = [point for cluster in self.bed_points for point in cluster]
        point_counts = Counter(map(tuple, all_data_points))
        most_repeated_points = point_counts.most_common()

        if not most_repeated_points:
            logging.info(self.Time_bed + '  Status >>>>>> ' + " Not_detected")
            self.In_bed = False
            self.check_in_bed_transition()
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
                self.check_in_bed_transition()
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
                logging.info("301 : " + self.Time_bed + ' -> ' + " In_bed")

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

    @profile(stream=open('VAL_mp_final_fall.log', 'w+'))
    def final_decision_for_fall(self, date):
        print("*******CALL final_decision_for_fall")
        final_decision = False

        while not final_decision:
            index_bed_sfd = self.find_bed_result_list_index(date)
            index = self.find_ml_result_list_index(date)
            print(f"^^^^^^^^^^^{index_bed_sfd} {index}")
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
        return
