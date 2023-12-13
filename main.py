import cv2
import speech_recognition as sr

from gtts import gTTS
from time import sleep
from playsound import playsound

from image_rec import ImageRec

def main() -> None:
    # Device Setup
    capture   = cv2.VideoCapture(0)
    voice_rec = sr.Recognizer()

    # Load Model Files
    config_file  = "resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model = "resources/frozen_inference_graph.pb"
    labels       = "resources/labels.txt"

    img_rec = ImageRec(config_file, frozen_model, labels)

    frame_count = 0 # Frames rendered over past 10 seconds
    while True:
        is_true, frame = capture.read()
        # Text Image Processing
        gry_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gry_frame, 145, 255, cv2.THRESH_BINARY)

        # Detection Image Processing
        class_indexes, confidence, bbox = img_rec.find_object(frame)

        # Speaking Recognized Objects
        if frame_count % 100 == 0:
            text = "I see "
            
            if len(class_indexes) > 0:
                obj_count = {}

                for idx in class_indexes.tolist():
                    if idx not in obj_count and idx <= 80:
                        obj_count[img_rec.class_labels[idx-1]] = class_indexes.tolist().count(idx)
                for key in obj_count:
                    text += f", {obj_count[key]} {key}"
                    if int(obj_count[key]) > 2:
                        text += "s"

            gTTS(text, lang='en', slow=False).save('temp/obj_count.mp3')
            playsound("temp/obj_count.mp3")
                                    
        # Show frame; break if q pressed
        cv2.imshow("Balls", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()