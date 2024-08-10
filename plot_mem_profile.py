import re
import matplotlib.pyplot as plt

WORK_DIR = '/home/h3trinh/clean_run'

def parse_memory_log(file_path):
    """
    Parses the memory profile log file to extract memory usage data.
    
    Args:
        file_path (str): Path to the memory profile log file.
    
    Returns:
        list: A list of memory usages.
    """
    memory_usage = []
    # Updated regex pattern to match lines containing @profile and memory usage
    memory_pattern = re.compile(r'\b(\d+.\d+) MiB\b')

    with open(file_path, 'r') as file:
        for line in file:
            if '@profile' in line:
                memory_match = memory_pattern.search(line)
                if memory_match:
                    memory = float(memory_match.group(1))
                    memory_usage.append(memory)

    return memory_usage

def plot_memory_usage(memory_usages, title):
    """
    Plots the memory usage over time.
    
    Args:
        memory_usages (list): A list of memory usages.
        title (str): Title of the plot.
    """
    times = list(range(len(memory_usages)))
    memories = memory_usages

    plt.figure(figsize=(12, 6))
    plt.plot(times, memories, marker='o')
    plt.xlabel('Profile Point')
    plt.ylabel('Memory Usage (MiB)')
    plt.title(title)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title}.png', bbox_inches='tight')
    plt.show()

def main():
    # File paths of the memory profile logs
    log_files = {
        'run_ml': f'{WORK_DIR}/mp_run_ml.log',
        'run_signal_processing_move': f'{WORK_DIR}/mp_run_move.log',
        #'process_file': f'{WORK_DIR}/VAL_mp_process_file.log',
        #'model_test': f'{WORK_DIR}/VALmp_model_test.log',
        'run_signal_processing_fall_detection': f'{WORK_DIR}/mp_run_fall.log',
        'Bed_Detection': f'{WORK_DIR}/mp_bed_detection.log',
        'run_benchmark2' : f'{WORK_DIR}/mp_run_bm2.log'
    }

    # Parse the memory usage logs and plot the data
    for function_name, file_path in log_files.items():
        memory_usage = parse_memory_log(file_path)
        plot_memory_usage(memory_usage, f'ONNX MODEL Memory Usage for {function_name} for 1 day')

if __name__ == '__main__':
    main()
