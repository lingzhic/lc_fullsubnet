import os
import csv
import wave

# The directory containing the .wav files
clean_dir = '/mnt/sam/Lingzhi/edinburgh/clean_trainset_28spk_wav'
noisy_dir = '/mnt/sam/Lingzhi/edinburgh/noisy_trainset_28spk_wav'

# The path to the CSV file where the information will be saved
csv_file_path = '/mnt/sam/Lingzhi/Git_repo/asteroid/egs/librimix/ConvTasNet_edb/edingburgh_meta/edingburgh_valid_meta.csv'

def get_wav_details(filepath):
    """
    Returns the length in seconds and the sample rate of the .wav file.
    """
    with wave.open(filepath, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        length = frames / float(rate)
        return length, rate

# Find all .wav files in the specified directory and get their details
wav_files_details = []
for root, dirs, files in os.walk(clean_dir):
    for file in files:
        if file.endswith('.wav'):
            clean_file_path = os.path.join(root, file)
            noisy_file_path = os.path.join(noisy_dir, file)
            length, rate = get_wav_details(clean_file_path)
            wav_files_details.append((file[:-4], clean_file_path, noisy_file_path, length, rate))

# Write the information to a CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Clean_filepath', 'Noisy_filepath', 'Length (s)', 'Sample Rate (Hz)'])  # Writing the header
    for details in wav_files_details:
        writer.writerow(details)

print(f'Done! Information about .wav files has been written to {csv_file_path}')
