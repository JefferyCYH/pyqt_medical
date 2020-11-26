import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from SecondWindow import MyApp, MyApp_LGE


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # 按钮
        self.button_switch = QPushButton('启动')
        self.button_quit = QPushButton('退出')

        # 下拉菜单
        self.demo_box = QComboBox(self)
        self.dictType = {0: 'PNG', 1: 'LGE', 2: 'SAX'}

        # 图像
        self.label_show_camera = QLabel()
        self.label_show_logo = QLabel()

        self.set_slot()  # 连接槽函数
        self.set_ui()  # 设置UI


    def quit(self):
        """关闭软件"""
        self.close()

    def set_slot(self):
        """连接槽函数"""
        self.button_switch.clicked.connect(self.switch)
        self.button_quit.clicked.connect(self.quit)


    def set_ui(self):
        """界面美化"""
        # 美化按钮
        for button in (self.button_switch, self.button_quit):
            button.setStyleSheet("QPushButton{background:#000000;"
                                 #"border:1px solid #DA251C;"
                                 "color:white;"
                                 "border-radius:5px;}"
                                 "QPushButton:hover{background:#E0E0E0;}"
                                 "QPushButton:pressed{background:#D0D0D0;}")
            button.setFixedSize(100, 30)

        # 美化下拉菜单
        self.demo_box.setFixedSize(100, 30)
        # self.demo_box.addItems(['PNG', 'LGE', '肾脏CT'])
        self.demo_box.addItems([self.dictType.get(0), self.dictType.get(1), self.dictType.get(2)])
        # self.demo_box.currentIndexChanged[str].connect(self.listchange)
        self.demo_box.setStyleSheet("QLabel{background:#000000;"
                                    "border:1px solid #DA251C;"
                                    "border-radius:5px;}")

        # 美化标题
        self.label_show_logo.setFixedSize(800, 160)
        img = cv2.cvtColor(cv2.imread('./LOGO.png'), cv2.COLOR_BGR2RGB)
        logo = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        self.label_show_logo.setPixmap(QPixmap.fromImage(logo))
        self.label_show_logo.setStyleSheet("QLabel{border:0px;}")

        # 界面布局
        buttons_layout = QGridLayout()
        buttons_layout.addWidget(self.demo_box, 0, 0, 1, 1)
        buttons_layout.addWidget(self.button_switch, 1, 0, 1, 1)
        buttons_layout.addWidget(self.button_quit, 2, 0, 1, 1)

        panel_layout = QHBoxLayout()
        panel_layout.addWidget(self.label_show_logo)
        panel_layout.addLayout(buttons_layout)


        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_show_camera)
        main_layout.addLayout(panel_layout)

        widget = QWidget()
        widget.setStyleSheet("QWidget{color:#FFFFFF;"
                             "background:#000000;"
                             "border:1px solid #CFCFCF;"
                             "border-radius:10px;}")
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlag(Qt.FramelessWindowHint)
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)


    def switch(self):
        if self.demo_box.currentText() == 'PNG':
            self.PNGwindow = MyApp()
            # MainWindow.close(self)
            self.PNGwindow.show()

        elif self.demo_box.currentText() == 'LGE':
            self.LGEwindow = MyApp_LGE()
            # MainWindow.close(self)
            self.LGEwindow.show()

        else:
            print('Wrong!')
            raise RuntimeError('Wrong!')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(open('./custom/styleSheet.qss', encoding='utf-8').read())
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    UI = MainWindow()
    UI.show()
    sys.exit(app.exec_())