from custom.tableWidget import *
from config import tables


class StackedWidget_LGE(QStackedWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        for table in tables:
            self.addWidget(table(parent=parent))
        self.setMinimumWidth(200)
