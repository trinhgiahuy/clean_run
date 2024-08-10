from src.processing.ProducerUpdate import ProducerUpdate
import threading
# DATA_FILE_PATH="/home/h3trinh/2022-08-27/0x/*.pickle"
DATA_FILE_PATH="/home/j63chow/data/*.pickle"
# DATA_FILE_PATH="/home/h3trinh/2022-08-27/03/*.pickle"

def main():
    # producer = ProducerDequeue(dataPath=DATA_FILE_PATH)
    # producer = ProcedureSingleTF(dataPath=DATA_FILE_PATH)
    producer = ProducerUpdate(dataPath=DATA_FILE_PATH)
    # producer = ProducerFirst(dataPath=DATA_FILE_PATH)
    # Spawn thread to run producer.run()

    # Key-metrics when we do benchmarking
    #   + Do we want to unzip the files all then run inferencing or 
    #   + [x] do we want to as long as start unzip file, we start doing computation
    
    # For prodcuer-consumer model, do we need to seperate
    


if __name__ == '__main__':
    main()
