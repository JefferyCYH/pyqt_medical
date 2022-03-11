from RAWprocess.tableWidget_RAW import *
from RAWprocess.config_RAW import tables



class StackedWidget_RAW(QStackedWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        for table in tables:
            self.addWidget(table(parent=parent))
        self.setMinimumWidth(200)
