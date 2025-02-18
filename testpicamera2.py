import cv2
from picamera2 import Picamera2

# Picamera2() 객체 생성
picam2 = Picamera2()
# preview 설정
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
# picam2 시작
picam2.start()

try:
    while True:
        # picam2.caputure_array() : picam2 객체 frame 이미지를 배열로 반환
        im = picam2.capture_array()
        cv2.imshow("Camera", im)  # 배열을 데이터를 출력

        # 's' 버튼 클릭시 이미지 캡쳐 저장
        key = cv2.waitKey(1)
        if key == ord('s'):
            # OpenCV 사용하여 이미지 저장
            cv2.imwrite("captured_image.jpg", im)
            print("Image saved!")

        # Exit the loop when 'q' is pressed
        elif key == ord('q'):
            break

finally:
    # 리소스 해제
    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()