# beep 소리 출력 GUI
from PyQt5.QtWidgets import *
import sys
# import winsound     # windows 내 사운드       * RBPi에서는 작동 X

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        # 윈도우(창) 이름과 위치 지정
        self.setWindowTitle('Beep sound')
        self.setGeometry(1200,800,500,100)       # setGeometry(ax: int, ay: int, aw: int, ah: int)

        # 버튼 생성
        shortBeepButton = QPushButton('Beep', self)
        longBeepButton = QPushButton('Beeeeep', self)
        quitButton = QPushButton('Quit', self)     
        self.label=QLabel('환영합니동', self)

        # 버튼 위치 지정
        shortBeepButton.setGeometry(10, 10, 100, 30)
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)

        # 콜백 함수 지정
        shortBeepButton.clicked.connect(self.shortBeepFunction)
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)

    def shortBeepFunction(self):
        self.label.setText('1000hz, 0.5s')
        # winsound.Beep(1000, 500)
    def longBeepFunction(self):
        self.label.setText('1000hz, 3s')
        # winsound.Beep(1000, 3000)
    def quitFunction(self):
        self.close()


app = QApplication(sys.argv)
win = BeepSound()
win.show()
app.exec_()