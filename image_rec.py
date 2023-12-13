import cv2

class ImageRec:
    def __init__(self, model_path: str, config_path: str, label_path: str) -> None:
        self.model = cv2.dnn.DetectionModel(model_path, config_path) # Loading DM Model
        self.class_labels = [label for label in open(label_path, 'rt').read().split('\n')] # Load labels.txt as list

        # Model Setup
        self.model.setInputSize(320, 320)
        self.model.setInputScale(1.0/127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)

    def find_object(self, frame) -> tuple:
        class_indexes, confidence, bbox = self.model.detect(frame, confThreshold=0.5)

        if len(class_indexes) != 0:
            # Framing Recognized Objects (dev)
            for class_ind, conf, boxes in zip(class_indexes.flatten(), confidence.flatten(), bbox):
                if class_ind <= 80:
                    cv2.rectangle(frame, boxes, (255,0,0), 1)
                    cv2.putText(frame, f"label: {self.class_labels[class_ind-1]}", (boxes[0]+10, boxes[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                    cv2.putText(frame, f"conf: {conf}", (boxes[0]+10, boxes[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        return (class_indexes, confidence, bbox)