# Reading the h5 file
data_file = h5py.File('fcsn_tvsum.h5')

# Predicting scores for a particular video using the model
pred_score = model.predict(np.array(data_file['video_30']['feature']).reshape(-1,320,1024))
video_info = data_file['video_30']
pred_score, pred_selected, pred_summary = select_keyshots(video_info, pred_score)


# Selected shots
print(pred_selected)


# Getting the output summary video
import cv2
cps = video_info['change_points'][()]

video = cv2.VideoCapture('video_30.mp4')
frames = []
success, frame = video.read()
while success:
    frames.append(frame)
    success, frame = video.read()
frames = np.array(frames)
keyshots = []
for sel in pred_selected:
    for i in range(cps[sel][0], cps[sel][1]):
         keyshots.append(frames[i])
keyshots = np.array(keyshots)

video_writer = cv2.VideoWriter('summary.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, keyshots.shape[2:0:-1])
for frame in keyshots:
    video_writer.write(frame)
video_writer.release()


