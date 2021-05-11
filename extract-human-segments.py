
import cvlib    # high level module, uses YOLO model with the find_common_objects method
import cv2      # image/video manipulation, allows us to pass frames to cvlib
from argparse import ArgumentParser
import os
import sys
from datetime import datetime
from tqdm import tqdm

# these will need to be fleshed out to not miss any formats
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.gif']
VID_EXTENSIONS = ['.mov', '.mp4', '.avi', '.mpg', '.mpeg', '.m4v', '.mkv']

# used to make sure we are at least examining one valid file
VALID_FILE_ALERT = False
# if an error is dectected, even once. Used for alerts
ERROR_ALERT = False
#used for alerts. True if human found once
HUMAN_DETECTED_ALERT = False

def process_frame_list(frame_list,frame_range,frame_count,fps,time_padding=10):
    processed_frame_list = []
    start_frame_ind = 0
    end_frame_ind = 0
    ind = 0
    frame_padding = fps*time_padding
    while end_frame_ind+1 < len(frame_list):
        if frame_list[end_frame_ind+1] - frame_list[end_frame_ind] == frame_range:
            end_frame_ind += 1
        else:
            start_frame= max(frame_list[start_frame_ind]-frame_padding,0)
            end_frame = min(frame_list[end_frame_ind]+frame_padding,frame_count)

            if len(processed_frame_list) > 0 and processed_frame_list[-1][1] >= start_frame - frame_padding:
                processed_frame_list[-1] = (processed_frame_list[-1][0],end_frame)
            else:
                processed_frame_list.append((start_frame,end_frame))
            start_frame_ind = end_frame_ind + 1
            end_frame_ind += 1
    
    return processed_frame_list


def remove_other_labels(bbox,labels,conf):
    for ind in range(len(labels)-1,-1,-1):
        if labels[ind] != "person":
            labels.pop(ind)
            bbox.pop(ind)
            conf.pop(ind)
    return bbox,labels,conf
    

def snip_video_segments_from_frame_list(frame_list,video_file_name,frame_count,save_directory,frame_range,time_padding,frame_dict):
    print("frame_list",frame_list)
    
    vid = cv2.VideoCapture(video_file_name)
    frame_width = int( vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int( vid.get(cv2.CAP_PROP_FPS))
    frame_height =int( vid.get( cv2.CAP_PROP_FRAME_HEIGHT))
    processed_frame_list = process_frame_list(frame_list,frame_range,frame_count,fps,time_padding)

    video_count = 1
    frame_count = 0
    frame_range_count = 1
    total_frame_range_count = len(processed_frame_list)
    video_file_path = os.path.join(save_directory,'processed_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(f'{video_file_path}', fourcc, fps, (frame_width, frame_height)) 
    print(f"Processed Frame Ranges {processed_frame_list}")

    for frame_range in processed_frame_list:
        print(f"Processing frame range {frame_range_count} of {total_frame_range_count}")
        for frame_number in tqdm(range(frame_range[0],frame_range[1]+1),"Creating video snippet from frame range found to contain humans"):
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            _ , frame = vid.read()
            try:
                height,width,channels = frame.shape
            except Exception as e:
                print(f"Error {e}, unable to read frame number {frame_number}")
                continue
            if frame_number in frame_dict:
                out.write(frame_dict[frame_number])
            else:
                out.write(frame)
        frame_range_count +=1
    out.release()
    print("video saved to",video_file_path)

        
    

# function takes a file name(full path), checks that file for human shaped objects
# saves the frames with people detected into directory named 'save_directory'
def humanChecker(video_file_name, save_directory, yolo='yolov4', nth_frame=20, confidence=0.65, gpu=False,time_padding=10,boxes_flag= False):
    # save_directory_2 = os.path.join()
    # for modifying our global variarble VALID_FILE
    global VALID_FILE_ALERT
    frame_dict = {}

    # tracking if we've found a human or not
    is_human_found = False
    analyze_error = False
    is_valid = False

    # we'll need to increment every time a person is detected for file naming
    person_detection_counter = 0

    # check if image
    if os.path.splitext(video_file_name)[1] in IMG_EXTENSIONS:
        frame = cv2.imread(video_file_name)  # our frame will just be the image
        #make sure it's a valid image
        if frame is not None:
            frame_count = 8   # this is necessary so our for loop runs below
            nth_frame = 1
            VALID_FILE_ALERT = True
            is_valid = True
            print(f'Image')
        else:
            is_valid = False
            analyze_error = True
            

    # check if video
    elif os.path.splitext(video_file_name)[1] in VID_EXTENSIONS:
        vid = cv2.VideoCapture(video_file_name)
        # get approximate frame count for video
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        #make sure it's a valid video
        if frame_count > 0:
            VALID_FILE_ALERT = True
            is_valid = True
            print(f'{frame_count} frames')
        else:
            is_valid = False
            analyze_error = True
    else:
        print(f'\nSkipping {video_file_name}')

    
    if is_valid:
        # look at every nth_frame of our video file, run frame through detect_common_objects
        # Increase 'nth_frame' to examine fewer frames and increase speed. Might reduce accuracy though.
        # Note: we can't use frame_count by itself because it's an approximation and could lead to errors
        frame_number_list = []
        for frame_number in tqdm(range(1, frame_count - 6, nth_frame),desc="Detecting people in frames using YOLO"):

            # if not dealing with an image
            if os.path.splitext(video_file_name)[1] not in IMG_EXTENSIONS:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, frame = vid.read()
                frame_dict[frame_number] = frame

            # feed our frame (or image) in to detect_common_objects
            try:
                
                bbox, labels, conf = cvlib.detect_common_objects(frame, model=yolo, confidence=confidence, enable_gpu=gpu)
            except Exception as e:
                print(e)
                analyze_error = True
                break

            if 'person' in labels:
                person_detection_counter += 1
                is_human_found = True

                # create image with bboxes showing people and then save
                bbox,labels,conf = remove_other_labels(bbox,labels,conf)
                marked_frame = cvlib.object_detection.draw_bbox(frame, bbox, labels, conf, write_conf=True)
                save_file_name = os.path.basename(os.path.splitext(video_file_name)[0]) + '-' + str(person_detection_counter) + '.jpeg'
                cv2.imwrite(save_directory + '/snapshots/' + save_file_name , marked_frame)
                frame_number_list.append(frame_number)
        if not boxes_flag:
            frame_dict = {}  

    if is_valid and person_detection_counter > 0:
        snip_video_segments_from_frame_list(frame_number_list,video_file_name,frame_count,save_directory,nth_frame,time_padding=time_padding,frame_dict=frame_dict)
    return is_human_found, analyze_error


# takes a directory and returns all files and directories within
def getListOfFiles(dir_name):
    list_of_files = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_files:
        # ignore hidden files and directories
        if entry[0] != '.':
            # Create full path
            full_path = os.path.join(dir_name, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(full_path):
                all_files = all_files + getListOfFiles(full_path)
            else:
                all_files.append(full_path)
    return all_files



#############################################################################################################################
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', default='', help='Path to video folder')
    parser.add_argument('-f', default='', help='Used to select an individual file')
    parser.add_argument('--tiny_yolo', action='store_true', help='Flag to indicate using YoloV4-tiny model instead of the full one. Will be faster but less accurate.')
    parser.add_argument('--confidence', type=int, choices=range(1,100), default=65, help='Input a value between 1-99. This represents the percent confidence you require for a hit. Default is 65')
    parser.add_argument('--frames', type=int, default=10, help='Only examine every nth frame. Default is 10')
    parser.add_argument('--gpu', action='store_true', help='Attempt to run on GPU instead of CPU. Requires Open CV compiled with CUDA enables and Nvidia drivers set up correctly.')
    parser.add_argument('--time_padding',type=int,default=10, help='number of extra seconds of footage to include on either side of an extracted video clip')
    parser.add_argument('--draw_boxes',action='store_true', help='Include bounding boxes and confidence in the processed video')
    args = vars(parser.parse_args())

    # decide which model we'll use, default is 'yolov3', more accurate but takes longer
    if args['tiny_yolo']:
        yolo_string = 'yolov4-tiny'
    else:
        yolo_string = 'yolov4'

        
    #check our inputs, can only use either -f or -d but must use one
    if args['f'] == '' and args['directory'] == '':
        print('You must select either a directory with -d <directory> or a file with -f <file name>')
        sys.exit(1)
    if args['f'] != '' and args['directory'] != '' :
        print('Must select either -f or -d but can''t do both')
        sys.exit(1)


    every_nth_frame = args['frames']
    
    confidence_percent = args['confidence'] / 100
    
    gpu_flag = False
    if args['gpu']:
        gpu_flag = True
    
    boxes_flag= False
    if args['draw_boxes']:
        boxes_flag = True
    
    time_padding = args['time_padding']


    print('Beginning Detection')
    print(f"Confidence threshold set to {args['confidence']}%")
    print(f'Examining every {every_nth_frame} frames.')
    print(f"GPU is set to {args['gpu']}")
    print('\n\n')

    if args['f'] == '':
        video_directory_list = getListOfFiles(args['directory'] + '/')
    else:
        video_directory_list = [args['f']]

    # what video we are on
    working_on_counter = 1

    for video_file in video_directory_list:
        print(f'Examining {video_file}: {working_on_counter} of {len(video_directory_list)}: {int((working_on_counter/len(video_directory_list)*100))}%    ', end='')
        custom_save_dir = f"../output/{video_file.split('/')[-1]}_processed"
        if not os.path.exists(custom_save_dir):
            if not os.path.exists("../output"):
                os.mkdir("../output")
                
            os.mkdir(custom_save_dir)
            os.mkdir(f"{custom_save_dir}/snapshots")
        else:
            print("skipping video as directory already exists")
            working_on_counter +=1
            continue
        # check for people
        human_detected, error_detected =  humanChecker(str(video_file), custom_save_dir, yolo=yolo_string, nth_frame=every_nth_frame, confidence=confidence_percent, gpu=gpu_flag,time_padding=time_padding,boxes_flag = boxes_flag)
            
        if human_detected:    
            HUMAN_DETECTED_ALERT = True
            print(f'At least one Human detected in {video_file}')
        
        if error_detected:
            ERROR_ALERT = True
            print(f'\nError in analyzing {video_file}')

        working_on_counter += 1

    if VALID_FILE_ALERT is False:
        print('No valid image or video files were examined')
