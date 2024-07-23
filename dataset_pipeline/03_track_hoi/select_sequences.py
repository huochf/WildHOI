import os
import json

good_sequences = []
need_to_split_sequences = []

for file in os.listdir('./good'):
    file_name = file.split('.')[0]
    _, _, video_id, hoi_idx = file_name.split('_')
    good_sequences.append((video_id, '{:03d}'.format(int(hoi_idx))))

# for file in os.listdir('./need_to_split'):
#     file_name = file.split('.')[0]
#     _, _, video_id, hoi_idx = file_name.split('_')
#     need_to_split_sequences.append((video_id, '{:03d}'.format(int(hoi_idx))))

with open('./tennis_sequences.json', 'w') as f:
    json.dump({'good': good_sequences, 'need_to_split': need_to_split_sequences}, f)
