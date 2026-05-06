from modules.video_capture import VideoCapture

def main():
    video = VideoCapture()

    while True:
        frame = video.get_frame()
        if frame is None:
            break

    video.release()

if __name__ == "__main__":
    main()